//! `victory list-diff`: compare two `victory translate --validate --list`
//! outputs program by program.
//!
//! Measuring a shader change on real data means running `translate --list`
//! before and after it and comparing the two listings. The report this
//! prints holds only counts and failure reasons, never a program's label
//! (labels carry the title's shader names), so it can be pasted into an
//! issue or a pull request as it is (see `docs/asset-policy.md`).

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};

/// The stage names `translate` prints after a program's label.
const STAGES: [&str; 8] = ["Ls", "Hs", "Es", "Gs", "Vs", "Ps", "Ms", "Cs"];

/// What one listing says about one program: every outcome line it has.
#[derive(Debug, Default, PartialEq)]
struct Outcome {
    validated: bool,
    /// Failure reasons, sorted (a program can fail in several passes).
    reasons: Vec<String>,
}

impl Outcome {
    /// The failure reasons as one key.
    fn why(&self) -> String {
        if self.reasons.is_empty() {
            // Emitted but neither validated nor refused: the listing was
            // made without `--validate`.
            "(no outcome line)".to_owned()
        } else {
            self.reasons.join("; ")
        }
    }
}

/// A program line's label without the offset of the failing instruction
/// (that offset changes when a fix moves the failure), and its outcome.
fn split(line: &str) -> Option<(&str, &str)> {
    let line = line.trim_start_matches('\u{feff}');
    if line.starts_with(char::is_whitespace) {
        return None;
    }
    let (label, outcome) = line.split_once("  ")?;
    let label = match label.rsplit_once(" at 0x") {
        Some((head, hex)) if !hex.is_empty() && hex.bytes().all(|b| b.is_ascii_hexdigit()) => head,
        _ => label,
    };
    let stage = label.rsplit_once(' ').map_or(label, |(_, s)| s);
    STAGES.contains(&stage).then(|| (label, outcome.trim_end()))
}

/// Every program a listing names, by label.
fn parse(text: &str) -> BTreeMap<&str, Outcome> {
    let mut out: BTreeMap<&str, Outcome> = BTreeMap::new();
    for (label, outcome) in text.lines().filter_map(split) {
        let o = out.entry(label).or_default();
        match outcome {
            "validated" => o.validated = true,
            "pipeline created" => {}
            reason => o.reasons.push(reason.to_owned()),
        }
    }
    for o in out.values_mut() {
        o.reasons.sort();
        o.reasons.dedup();
    }
    out
}

#[derive(Debug, Default, PartialEq)]
struct Diff {
    compared: usize,
    only_before: usize,
    only_after: usize,
    validated_before: usize,
    validated_after: usize,
    /// Programs that validate only after, by their failure before.
    gained: BTreeMap<String, usize>,
    /// Programs that validate only before, by their failure after.
    lost: BTreeMap<String, usize>,
    /// Programs that fail in both, for a different reason: (before, after).
    moved: BTreeMap<(String, String), usize>,
}

fn diff(before: &BTreeMap<&str, Outcome>, after: &BTreeMap<&str, Outcome>) -> Diff {
    let mut d = Diff::default();
    for (label, a) in before {
        let Some(b) = after.get(label) else {
            d.only_before += 1;
            continue;
        };
        d.compared += 1;
        d.validated_before += usize::from(a.validated);
        d.validated_after += usize::from(b.validated);
        match (a.validated, b.validated) {
            (false, true) => *d.gained.entry(a.why()).or_default() += 1,
            (true, false) => *d.lost.entry(b.why()).or_default() += 1,
            (false, false) if a.reasons != b.reasons => {
                *d.moved.entry((a.why(), b.why())).or_default() += 1;
            }
            _ => {}
        }
    }
    d.only_after = after.keys().filter(|l| !before.contains_key(*l)).count();
    d
}

/// Most frequent first; ties by reason, so two runs print the same report.
fn ranked<K: Clone + Ord>(m: &BTreeMap<K, usize>) -> Vec<(K, usize)> {
    let mut v: Vec<(K, usize)> = m.iter().map(|(k, n)| (k.clone(), *n)).collect();
    v.sort_by(|x, y| y.1.cmp(&x.1).then_with(|| x.0.cmp(&y.0)));
    v
}

fn report(d: &Diff, top: usize) -> String {
    let total = |m: &BTreeMap<String, usize>| m.values().sum::<usize>();
    let mut s = String::new();
    let mut line = |t: String| {
        s.push_str(&t);
        s.push('\n');
    };
    line(format!(
        "programs     {} in both ({} only before, {} only after)",
        d.compared, d.only_before, d.only_after
    ));
    line(format!(
        "validated    {} before, {} after ({:+})",
        d.validated_before,
        d.validated_after,
        d.validated_after as i64 - d.validated_before as i64
    ));
    line(format!(
        "gained       {}, lost {}",
        total(&d.gained),
        total(&d.lost)
    ));
    for (title, m) in [
        ("gained, by the failure before", &d.gained),
        ("lost, by the failure after", &d.lost),
    ] {
        if !m.is_empty() {
            line(format!("{title}:"));
            for (why, n) in ranked(m) {
                line(format!("  {n:>7}  {why}"));
            }
        }
    }
    if !d.moved.is_empty() {
        let moved = ranked(&d.moved);
        let n: usize = moved.iter().map(|m| m.1).sum();
        line(format!("failing in both, reason changed ({n}):"));
        for ((from, to), n) in moved.iter().take(top) {
            line(format!("  {n:>7}  {from}  ->  {to}"));
        }
        if moved.len() > top {
            line(format!("  ... {} more pairs (--top)", moved.len() - top));
        }
    }
    s
}

fn read_listing(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path)?;
    // Windows PowerShell 5.1 redirects native stdout as UTF-16LE with a BOM.
    let Some(body) = bytes.strip_prefix(&[0xff, 0xfe]) else {
        return Ok(String::from_utf8(bytes)?);
    };
    let (units, tail) = body.as_chunks::<2>();
    anyhow::ensure!(tail.is_empty(), "truncated UTF-16LE code unit");
    let units = units.iter().copied().map(u16::from_le_bytes);
    Ok(char::decode_utf16(units).collect::<Result<String, _>>()?)
}

pub(crate) fn run(before: &Path, after: &Path, top: usize) -> Result<()> {
    let read = |p: &Path| read_listing(p).with_context(|| format!("reading {}", p.display()));
    let (a, b) = (read(before)?, read(after)?);
    let (a, b) = (parse(&a), parse(&b));
    anyhow::ensure!(
        !a.is_empty() && !b.is_empty(),
        "no program lines found: make both listings with `victory translate --validate --list`"
    );
    print!("{}", report(&diff(&a, &b), top));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const BEFORE: &str = "\u{feff}validator    C:\\sdk\\dxil.dll
containers   3 (0 duplicates skipped)
x.rpf@0x10 name_a Vs  validated
x.rpf@0x20 name_b Ps at 0x1f4  emit: a lane mask read as data
x.rpf@0x30 name_c Ms at 0x8  emit: a 64-bit image atomic
x.rpf@0x40 name_d Cs  validator: error: one.  See other errors for details.
x.rpf@0x50 name_e Ps at 0x10  I/O: an interpolation
x.rpf@0x50 name_e Ps at 0x24  resources: an address
x.rpf@0x60 gone Cs  validated
pair a + b: pipeline created
        2  emit: a lane mask read as data  (e.g. x.rpf@0x20 name_b Ps at 0x1f4)
";

    const AFTER: &str = "x.rpf@0x10 name_a Vs  emit: barycentrics read directly
x.rpf@0x20 name_b Ps  validated
x.rpf@0x30 name_c Ms at 0x9c  emit: a descriptor read only as data
x.rpf@0x40 name_d Cs  validator: error: one.  See other errors for details.
x.rpf@0x50 name_e Ps  validated
x.rpf@0x50 name_e Ps  pipeline created
x.rpf@0x70 new Gs  validated
";

    #[test]
    fn a_failure_offset_is_not_part_of_the_label() {
        assert_eq!(
            split("x.rpf@0x20 n Ps at 0x1f4  emit: why"),
            Some(("x.rpf@0x20 n Ps", "emit: why"))
        );
        assert_eq!(
            split("p@0x1 n Cs  validator: a.  See b."),
            Some(("p@0x1 n Cs", "validator: a.  See b."))
        );
        // Summary, header and pair lines are not programs.
        assert_eq!(split("        2  emit: x  (e.g. p@0x1 n Ps)"), None);
        assert_eq!(split("validator    C:\\dxil.dll"), None);
        assert_eq!(split("pair a + b: pipeline created"), None);
        // A label ending in " at 0x" without hex digits keeps it.
        assert_eq!(
            split("p@0x1 at 0x Ps  validated"),
            Some(("p@0x1 at 0x Ps", "validated"))
        );
    }

    #[test]
    fn several_failures_of_one_program_are_one_outcome() {
        let p = parse(BEFORE);
        assert_eq!(p.len(), 6);
        assert_eq!(
            p["x.rpf@0x50 name_e Ps"],
            Outcome {
                validated: false,
                reasons: vec![
                    "I/O: an interpolation".to_owned(),
                    "resources: an address".to_owned()
                ],
            }
        );
    }

    #[test]
    fn programs_are_compared_one_by_one() {
        let (a, b) = (parse(BEFORE), parse(AFTER));
        let d = diff(&a, &b);
        assert_eq!((d.compared, d.only_before, d.only_after), (5, 1, 1));
        assert_eq!((d.validated_before, d.validated_after), (1, 2));
        assert_eq!(
            d.gained,
            BTreeMap::from([
                ("emit: a lane mask read as data".to_owned(), 1),
                ("I/O: an interpolation; resources: an address".to_owned(), 1),
            ])
        );
        assert_eq!(
            d.lost,
            BTreeMap::from([("emit: barycentrics read directly".to_owned(), 1)])
        );
        // name_d fails the same way in both and is not listed as moved.
        assert_eq!(
            d.moved,
            BTreeMap::from([(
                (
                    "emit: a 64-bit image atomic".to_owned(),
                    "emit: a descriptor read only as data".to_owned()
                ),
                1
            )])
        );
    }

    #[test]
    fn the_report_names_no_program() {
        let r = report(&diff(&parse(BEFORE), &parse(AFTER)), 10);
        assert!(r.contains("validated    1 before, 2 after (+1)"), "{r}");
        assert!(r.contains("gained       2, lost 1"), "{r}");
        for label in ["name_", "x.rpf", "@0x", "gone", "new"] {
            assert!(!r.contains(label), "{label} in\n{r}");
        }
    }

    #[test]
    fn the_moved_list_is_cut_at_top() {
        let d = Diff {
            moved: BTreeMap::from([
                (("a".to_owned(), "b".to_owned()), 3),
                (("c".to_owned(), "d".to_owned()), 2),
            ]),
            ..Diff::default()
        };
        let r = report(&d, 1);
        assert!(r.contains("failing in both, reason changed (5):"), "{r}");
        assert!(r.contains("        3  a  ->  b"), "{r}");
        assert!(!r.contains("c  ->  d"), "{r}");
        assert!(r.contains("... 1 more pairs (--top)"), "{r}");
        // A --top above the number of pairs prints them all, with no note.
        let r = report(&d, 5);
        assert!(r.contains("c  ->  d") && !r.contains("more pairs"), "{r}");
    }

    #[test]
    fn windows_powershell_listings_work_in_either_position() {
        let temp = std::env::temp_dir();
        let pid = std::process::id();
        let utf8 = temp.join(format!("victory-listdiff-utf8-{pid}.txt"));
        let utf16 = temp.join(format!("victory-listdiff-utf16le-{pid}.txt"));
        let listing = "synthetic Vs  emit: unknown symbol 𝛼\r\n";
        std::fs::write(&utf8, format!("\u{feff}{listing}")).unwrap();
        let mut bytes = vec![0xff, 0xfe];
        bytes.extend(listing.encode_utf16().flat_map(u16::to_le_bytes));
        std::fs::write(&utf16, bytes).unwrap();
        assert_eq!(read_listing(&utf16).unwrap(), listing);
        run(&utf8, &utf16, 1).unwrap();
        run(&utf16, &utf8, 1).unwrap();

        for bytes in [&[0xff, 0xfe, 0x61][..], &[0xff, 0xfe, 0x00, 0xd8], &[0x80]] {
            std::fs::write(&utf16, bytes).unwrap();
            let err = run(&utf8, &utf16, 1).unwrap_err();
            assert_eq!(err.to_string(), format!("reading {}", utf16.display()));
        }
        let _ = (std::fs::remove_file(utf8), std::fs::remove_file(utf16));
    }

    #[test]
    fn a_listing_without_program_lines_is_refused() {
        // Made without --list: only the header and the summary.
        assert!(
            parse("programs     3 translated of 3\n        2  emit: x  (e.g. p@0x1 n Ps)\n")
                .is_empty()
        );
        let dir = std::env::temp_dir();
        let (empty, full) = (
            dir.join("victory-listdiff-empty.txt"),
            dir.join("victory-listdiff-full.txt"),
        );
        std::fs::write(&empty, "containers   0\n").unwrap();
        std::fs::write(&full, BEFORE).unwrap();
        let err = run(&empty, &full, 20).unwrap_err().to_string();
        assert!(err.contains("no program lines"), "{err}");
        let _ = (std::fs::remove_file(empty), std::fs::remove_file(full));
    }
}
