//! Host descriptor heaps behind the title's shader-visible heaps (decision
//! 0007, step 3). Translated shaders index the heaps directly at the
//! title's descriptor positions, so each host heap keeps the title heap's
//! order: slot `i` is made from what the title wrote at slot `i`, when it
//! writes it (view creation and descriptor copies), when a heap is synced for
//! direct CPU copies inside heap memory, and again when a draw reads it
//! through a table and finds the slot changed. Draws that index
//! `ResourceDescriptorHeap` also refresh the live buffer ranges tracked here
//! ([`HostHeap::refresh_bindless_buffers`]) so heap-indexed views stay as
//! current as table ones.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::{VIEW_TAG, View, view_at};
use crate::host::{Host, HostList, host};
use crate::shadow::Shadow;

/// Submission-time refresh state for a slot whose exact backing shadow may
/// differ from `buffer_range` (a structured alias, or a view off an element of
/// its host copy that reads its own aligned copy).
#[derive(Clone)]
enum SlotRefresh {
    Shadow {
        shadow: Arc<Shadow>,
        address: u64,
        len: u64,
    },
}

/// Title resource and host `(MostDetailedMip, MipLevels)` a slot's SRV was
/// made with after resident-mip narrowing.
type SrvMips = (usize, (u32, u32));

/// Maximum number of recent texture-descriptor writes retained in
/// [`TexTracker::changes`] for incremental bindless sync across `tex_gen`
/// bumps; overflowing falls back to a full `live_textures` walk once.
const MAX_TEX_CHANGES: usize = 4096;

/// Locks a heap's slots are spread over: slots in different stripes are
/// remade at the same time ([`HostHeap::mirror`]).
const SLOT_STRIPES: usize = 64;

/// Live title-memory buffer range a heap slot views, for bindless refresh.
/// The bool is whether the view is unordered (refresh must leave the host
/// copy in unordered-access state, not generic-read).
type BufRange = (Arc<Shadow>, u64, u64, bool);

/// Tracks which title slots hold buffer / constant views whose host copies
/// need a submission-time refresh when a shader indexes `ResourceDescriptorHeap`.
/// Unaligned and aliased structured slots stay in [`HostHeap::refreshes`] instead.
#[derive(Default)]
struct BufTracker {
    /// Title slot → host copy range. Dynamic slots are never recorded.
    live: HashMap<u32, BufRange>,
}

/// Tracks which live host textures each view-heap slot currently references,
/// keeping `O(active)` live sets of SRV and UAV resources for bindless heaps.
/// Every slot entry is keyed by `(resource, is_uav)` in `res_slots` so
/// `forget_resource` clears only the exact slots that held the released
/// resource (ABA-safe if the resource pointer is later reused).
#[derive(Default)]
struct TexTracker {
    generation: u64,
    /// Generation of the last evicted entry from `changes`; any caller whose
    /// `since_gen < oldest_gen` must perform a full `live_textures` walk.
    oldest_gen: u64,
    /// Bounded ring of `(generation, resource)` additions/overwrites so
    /// `bindless_textures` can sync newly written descriptors incrementally.
    changes: VecDeque<(u64, usize)>,
    slot_res: Vec<Option<(usize, bool, u32)>>,
    live_srvs: HashMap<usize, u32>,
    live_uavs: HashMap<usize, u32>,
    res_slots: HashMap<(usize, bool), HashSet<u32>>,
}

impl TexTracker {
    fn first_mip(&self, resource: usize) -> u32 {
        self.res_slots[&(resource, false)]
            .iter()
            .filter_map(|&slot| self.slot_res[slot as usize].map(|(_, _, mip)| mip))
            .min()
            .unwrap_or(0)
    }

    fn new(sampler: bool, count: u32) -> Self {
        Self {
            generation: 0,
            oldest_gen: 0,
            changes: VecDeque::new(),
            slot_res: if sampler {
                Vec::new()
            } else {
                vec![None; count as usize]
            },
            live_srvs: HashMap::new(),
            live_uavs: HashMap::new(),
            res_slots: HashMap::new(),
        }
    }

    fn record_change(&mut self, res: usize) {
        if self.changes.len() == MAX_TEX_CHANGES
            && let Some((evicted_gen, _)) = self.changes.pop_front()
        {
            self.oldest_gen = evicted_gen;
        }
        self.changes.push_back((self.generation, res));
    }

    fn add_entry(&mut self, slot: u32, res: usize, is_uav: bool) {
        let counts = if is_uav {
            &mut self.live_uavs
        } else {
            &mut self.live_srvs
        };
        *counts.entry(res).or_insert(0) += 1;
        self.res_slots
            .entry((res, is_uav))
            .or_default()
            .insert(slot);
    }

    fn remove_entry(&mut self, slot: u32, res: usize, is_uav: bool) {
        let counts = if is_uav {
            &mut self.live_uavs
        } else {
            &mut self.live_srvs
        };
        if let std::collections::hash_map::Entry::Occupied(mut e) = counts.entry(res) {
            if *e.get() <= 1 {
                e.remove();
            } else {
                *e.get_mut() -= 1;
            }
        }
        if let std::collections::hash_map::Entry::Occupied(mut e) =
            self.res_slots.entry((res, is_uav))
        {
            e.get_mut().remove(&slot);
            if e.get().is_empty() {
                e.remove();
            }
        }
    }
}

/// Backing host resource held by a descriptor slot in [`HostHeap::kept`].
///
/// Buffer descriptors backed by a [`Shadow`] retain the `Arc<Shadow>` itself
/// (not just its `ID3D12Resource`) so [`crate::shadow`] cannot evict a loose
/// shadow window while a live descriptor slot still names its buffer.
enum KeptBacking {
    Resource { _resource: ID3D12Resource },
    Shadow { _shadow: Arc<Shadow> },
}

type LiveTextures = (
    u64,
    Vec<(crate::object::Held, u32)>,
    Vec<crate::object::Held>,
);

pub(crate) struct HostHeap {
    pub(crate) heap: ID3D12DescriptorHeap,
    cpu: usize,
    gpu: u64,
    step: u32,
    /// Slots mirroring the title's; `spare` more follow them.
    count: u32,
    /// Dynamic slots past the title's: [`DYNAMIC`] in a view heap,
    /// [`SAMPLER_DYNAMIC`] in a capped sampler heap, none in another.
    spare: u32,
    sampler: bool,
    /// The record (view index and generation) each slot was last made from.
    /// Read without a lock (the scan in [`Self::sync_slots`] runs on every
    /// bind of every recording thread); written under the slot's stripe.
    made: Box<[AtomicU64]>,
    /// Serialize remaking one slot ([`Self::mirror`]) while other slots are
    /// made at the same time: one lock for the whole heap made every
    /// recording thread wait for every other's view creation.
    stripes: Box<[Mutex<()>]>,
    /// The host resource (or [`Shadow`]) each slot views, held while the slot
    /// does: shaders index heaps directly, so a slot the title has not
    /// rewritten since it released the resource may still be read, and a freed
    /// host resource read there is a GPU page fault (the console reads stale
    /// memory). Retaining `Arc<Shadow>` also keeps loose shadow windows pinned
    /// in the shadow cache while any descriptor slot references their buffer.
    kept: Mutex<Vec<Option<KeptBacking>>>,
    /// Slots requiring submission-time title-memory refresh (aliased structured
    /// SRV shadows, or a view's own copy when off an element of its host copy).
    refreshes: Mutex<HashMap<u32, SlotRefresh>>,
    /// Mirrors `refreshes.len()` so `refresh_slot` and `keep` avoid locking
    /// `refreshes` when empty.
    refresh_count: AtomicUsize,
    /// Host SRV mip windows `(MostDetailedMip, MipLevels)` after narrowing to
    /// resident levels, by slot. Used to remake the view when more streamed
    /// mips arrive (#616).
    srv_mips: Mutex<HashMap<u32, SrvMips>>,
    /// Where the search for free dynamic slots starts, and which are taken
    /// (by lists not yet reset).
    next: Mutex<(u32, Vec<bool>)>,
    /// Leaf-only tracker of live SRV/UAV textures across title slots.
    tex_tracker: Mutex<TexTracker>,
    /// Title slots holding buffer / constant views for bindless refresh.
    buf_tracker: Mutex<BufTracker>,
    /// Mirrors `tex_tracker.generation` so `tex_gen()` is lock-free on every
    /// bindless draw.
    tex_gen: AtomicU64,
}

/// Host heaps' GPU ranges (start, bytes, bytes per slot, sampler), for
/// device-removed reports: a shader indexing a heap past its end faults.
static GPU_RANGES: Mutex<Vec<(u64, u64, u64, bool)>> = Mutex::new(Vec::new());

/// For device-removed reports: where host GPU address `va` lies against
/// the host descriptor heaps.
pub(crate) fn near_heaps(va: u64) -> String {
    let ranges = GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner());
    let mut text = String::new();
    for &(start, bytes, step, sampler) in ranges.iter() {
        let kind = if sampler { "sampler" } else { "view" };
        let place = if va < start {
            format!("{:#x} below", start - va)
        } else if va < start + bytes {
            format!("inside, slot {}", (va - start) / step)
        } else {
            format!(
                "{:#x} past its end (slot {})",
                va - start - bytes,
                (va - start) / step
            )
        };
        text.push_str(&format!("; {kind} heap {start:#x}+{bytes:#x}: {place}"));
    }
    text
}

/// The host's limit for shader-visible view heaps on resource binding
/// tiers 1 and 2; tier 3 hardware may take more.
const PORTABLE_VIEWS: u32 = 1_000_000;

/// The largest view heap made: translated shaders send indices at or past
/// it to slot 0, so a larger heap would lose its top slots to them.
const MAX_VIEWS: u32 = victory_shader::emit::VIEW_HEAP_LIMIT;
const _: () = assert!(PORTABLE_VIEWS <= MAX_VIEWS && DYNAMIC < PORTABLE_VIEWS);
const _: () = assert!(SAMPLERS <= victory_shader::emit::SAMPLER_HEAP_LIMIT);

/// Slots past the title's in a view heap, for tables a draw needs made
/// afresh (constant buffers copied per draw). A list holds the slots it
/// took until it is reset (its submissions have run) or released.
/// 2^16 ran out seconds into the world (every world run on an RX 9070 XT,
/// each a device removal shortly after; #59). A heap the host caps (at
/// [`PORTABLE_VIEWS`], NVIDIA) gets as many, taken from the top of the
/// title's slots: with fewer the dynamic slots ran out there too (2^16 and
/// 2^17 both did, in a frame asking for up to 16,000 tables), and a draw
/// whose table is refused is dropped; the title's highest slot seen, about
/// 720,000, stays below the 737,000 that are left.
const DYNAMIC: u32 = 1 << 18;

/// The host's limit for shader-visible sampler heaps, on every binding
/// tier.
const SAMPLERS: u32 = D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE;

/// Dynamic slots of a sampler heap capped at [`SAMPLERS`], for tables
/// reaching past the title slots it keeps.
const SAMPLER_DYNAMIC: u32 = 256;

impl HostHeap {
    pub(crate) fn new(sampler: bool, count: u32) -> Option<Self> {
        let host = host()?;
        let kind = if sampler {
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER
        } else {
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
        };
        let make = |count: u32| -> windows::core::Result<ID3D12DescriptorHeap> {
            // SAFETY: plain heap creation.
            unsafe {
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: kind,
                        NumDescriptors: count,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })
            }
        };
        if sampler && count > SAMPLERS {
            // shortcut: samplers past the host's limit are left out; tables
            // reaching them are made afresh in the dynamic slots.
            tracing::warn!(count, "host sampler heap capped at {SAMPLERS}");
            let heap = match make(SAMPLERS) {
                Ok(heap) => heap,
                Err(e) => {
                    tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                    return None;
                }
            };
            return Self::with(
                host,
                heap,
                kind,
                SAMPLERS - SAMPLER_DYNAMIC,
                SAMPLER_DYNAMIC,
            );
        }
        if sampler {
            return match make(count) {
                Ok(heap) => Self::with(host, heap, kind, count, 0),
                Err(e) => {
                    tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                    None
                }
            };
        }
        // The title slots that fit under the largest heap made beside the
        // dynamic ones.
        let count_fitting = count.min(MAX_VIEWS - DYNAMIC);
        if count_fitting < count {
            // shortcut: as below, descriptors past the limit are left out.
            tracing::warn!(count, "host view heap capped at {MAX_VIEWS}");
        }
        let (heap, count) = match make(count_fitting + DYNAMIC) {
            Ok(heap) => (heap, count_fitting),
            Err(e) if count_fitting + DYNAMIC > PORTABLE_VIEWS => {
                // shortcut: descriptors past the host's limit are left out;
                // a title that reaches them needs remapping.
                tracing::warn!(
                    error = %e,
                    count = count_fitting,
                    "host view heap capped at {PORTABLE_VIEWS}"
                );
                (make(PORTABLE_VIEWS).ok()?, PORTABLE_VIEWS - DYNAMIC)
            }
            Err(e) => {
                tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                return None;
            }
        };
        Self::with(host, heap, kind, count, DYNAMIC)
    }

    /// All slots are allocated dynamically and held by recordings.
    pub(super) fn root_views() -> Option<Self> {
        const SLOTS: u32 = 4096;
        let host = host()?;
        let kind = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        // SAFETY: a shader-visible view heap of a supported size.
        let heap = unsafe {
            host.device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: SLOTS,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
        }
        .ok()?;
        Self::with(host, heap, kind, 0, SLOTS)
    }

    /// The host heap over `heap`: `count` title slots, then `extra` dynamic
    /// ones, all made null.
    fn with(
        host: &Host,
        heap: ID3D12DescriptorHeap,
        kind: D3D12_DESCRIPTOR_HEAP_TYPE,
        count: u32,
        extra: u32,
    ) -> Option<Self> {
        let sampler = kind == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
        // SAFETY: a live heap.
        let (cpu, step) = unsafe {
            (
                heap.GetCPUDescriptorHandleForHeapStart().ptr,
                host.device.GetDescriptorHandleIncrementSize(kind),
            )
        };
        tracing::debug!(count, sampler, "host descriptor heap");
        // SAFETY: a live shader-visible heap.
        let gpu = unsafe { heap.GetGPUDescriptorHandleForHeapStart().ptr };
        GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner()).push((
            gpu,
            u64::from(count + extra) * u64::from(step),
            u64::from(step),
            sampler,
        ));
        let h = HostHeap {
            heap,
            cpu,
            gpu,
            step,
            count,
            spare: extra,
            sampler,
            made: (0..count).map(|_| AtomicU64::new(0)).collect(),
            stripes: (0..SLOT_STRIPES).map(|_| Mutex::new(())).collect(),
            kept: Mutex::new((0..(count + extra)).map(|_| None).collect()),
            refreshes: Mutex::new(HashMap::new()),
            refresh_count: AtomicUsize::new(0),
            srv_mips: Mutex::new(HashMap::new()),
            next: Mutex::new((0, vec![false; extra as usize])),
            tex_tracker: Mutex::new(TexTracker::new(sampler, count)),
            buf_tracker: Mutex::new(BufTracker::default()),
            tex_gen: AtomicU64::new(0),
        };
        // A slot never written is undefined: shaders reading one (a slot
        // the title wrote in a way we do not know, or a table entry no
        // used-resource entry names) read a wild address. Every slot starts
        // null instead.
        let started = std::time::Instant::now();
        if let Err(e) = h.fill_null(kind, count + extra) {
            tracing::warn!(error = %e, "host descriptor heap not made null");
            return None;
        }
        tracing::debug!(
            ms = started.elapsed().as_millis(),
            "host descriptor heap made null"
        );
        Some(h)
    }

    /// Makes the first `total` slots null: one null slot in a CPU-side heap,
    /// doubled by copies to a block, the block copied over the heap.
    fn fill_null(&self, kind: D3D12_DESCRIPTOR_HEAP_TYPE, total: u32) -> windows::core::Result<()> {
        let Some(host) = host() else { return Ok(()) };
        let block_count = total.min(if self.sampler {
            D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE
        } else {
            1 << 16
        });
        // SAFETY: plain heap creation and copies inside live heaps.
        unsafe {
            let block: ID3D12DescriptorHeap =
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: kind,
                        NumDescriptors: block_count,
                        ..Default::default()
                    })?;
            let start = block.GetCPUDescriptorHandleForHeapStart();
            let at = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: start.ptr + (i * self.step) as usize,
            };
            if self.sampler {
                null_sampler(start);
            } else {
                null_view(start);
            }
            let mut filled = 1;
            while filled < block_count {
                let n = filled.min(block_count - filled);
                host.device
                    .CopyDescriptorsSimple(n, at(filled), start, kind);
                filled += n;
            }
            let mut done = 0;
            while done < total {
                let n = block_count.min(total - done);
                host.device
                    .CopyDescriptorsSimple(n, self.cpu(done), start, kind);
                done += n;
            }
        }
        Ok(())
    }

    /// A plain point sampler at slot `index`, for sampler slots without a
    /// view.
    fn null_sampler(&self, index: u32) {
        null_sampler(self.cpu(index));
    }
}

/// A plain point sampler at `at`.
fn null_sampler(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
        ..Default::default()
    };
    // SAFETY: a slot of a sampler heap of ours.
    unsafe { host.device.CreateSampler(&desc, at) };
}

impl HostHeap {
    /// Total dynamic slot capacity of this heap (independent of current free
    /// space), so callers can reject impossible table sizes before scanning.
    pub(crate) fn dynamic_capacity(&self) -> u32 {
        self.spare
    }

    /// `n` consecutive free dynamic slots (view heaps and capped sampler
    /// heaps), taken until [`HostHeap::release`]: the first's index.
    pub(crate) fn dynamic(&self, n: u32) -> Option<u32> {
        if n == 0 || self.spare == 0 {
            return None;
        }
        if n > self.spare {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(
                    n,
                    spare = self.spare,
                    sampler = self.sampler,
                    "a table larger than the dynamic slots is not made"
                );
            });
            return None;
        }
        let spare = self.spare as usize;
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let (next, taken) = &mut *guard;
        let n = n as usize;
        let found = super::find_free_run(taken, *next as usize, n);
        let Some(first) = found else {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(n, "no free dynamic descriptor slots: a table is not made")
            });
            return None;
        };
        taken[first..first + n].fill(true);
        *next = ((first + n) % spare) as u32;
        Some(self.count + first as u32)
    }

    /// The title slots this heap has host slots for; fewer than the title's
    /// when the host caps its size.
    pub(crate) fn count(&self) -> u32 {
        self.count
    }

    /// Gives back `n` dynamic slots from `first` (an index [`dynamic`]
    /// returned).
    ///
    /// [`dynamic`]: HostHeap::dynamic
    pub(crate) fn release(&self, first: u32, n: u32) {
        let Some(at) = first.checked_sub(self.count) else {
            return;
        };
        let end = at.saturating_add(n).min(self.spare) as usize;
        let slot_start = first as usize;
        let slot_end = self.count as usize + end;
        let retired: Vec<KeptBacking> = {
            let mut slots = self.kept.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(slice) = slots.get_mut(slot_start..slot_end) {
                slice.iter_mut().filter_map(Option::take).collect()
            } else {
                Vec::new()
            }
        };
        {
            let mut map = self.refreshes.lock().unwrap_or_else(|e| e.into_inner());
            if !map.is_empty() {
                for index in slot_start..slot_end {
                    map.remove(&(index as u32));
                }
            }
        }
        {
            let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(slots) = guard.1.get_mut(at as usize..end) {
                slots.fill(false);
            }
        }
        for old in retired {
            crate::host::retire(old);
        }
    }

    /// Makes slot `index` (a dynamic one) from `view`; a null view if the
    /// host cannot.
    pub(crate) fn make_at(&self, index: u32, view: &View) -> bool {
        let kept = make(view, self.cpu(index));
        if kept.is_none() {
            if self.sampler {
                self.null_sampler(index);
            } else {
                null_for(view, self.cpu(index));
            }
        }
        let made = kept.is_some();
        self.keep(index, kept.unwrap_or(Kept::Empty));
        made
    }

    /// Makes slot `index` (a dynamic one) a structured buffer alias with
    /// element width `stride` without mutating the canonical title heap slot.
    pub(crate) fn make_structured_at(&self, index: u32, view: &View, stride: u32) -> bool {
        let Some((address, len, desc)) = buffer_srv_parts(view) else {
            return false;
        };
        let Some(alias) = structured_alias(desc, stride) else {
            return false;
        };
        let Some(kept) = buffer_srv_inner(address, len, alias, self.cpu(index), true) else {
            return false;
        };
        self.keep(index, kept);
        true
    }

    #[cfg(test)]
    pub(crate) fn gpu(&self, index: u32) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        self.gpu_handle(index)
    }

    /// Makes slot `index` read nothing: a null view, or a plain sampler.
    pub(crate) fn make_null(&self, index: u32) {
        if self.sampler {
            self.null_sampler(index);
        } else {
            null_view(self.cpu(index));
        }
        self.keep(index, Kept::Empty);
        self.track_buffer(index, None);
    }

    /// Slot `index` now views `kept` (or nothing of the host's).
    fn keep(&self, index: u32, kept: Kept) {
        let (resource, refresh, host_mips) = match kept {
            Kept::Empty => (None, None, None),
            Kept::Resource(r) => (Some(KeptBacking::Resource { _resource: r }), None, None),
            Kept::Shadow(s) => (Some(KeptBacking::Shadow { _shadow: s }), None, None),
            Kept::Texture {
                resource,
                title,
                is_uav: false,
                host_mips: Some(mips),
            } => (
                Some(KeptBacking::Resource {
                    _resource: resource,
                }),
                None,
                Some((title, mips)),
            ),
            Kept::Texture { resource, .. } => (
                Some(KeptBacking::Resource {
                    _resource: resource,
                }),
                None,
                None,
            ),
            Kept::AliasedShadow {
                shadow,
                address,
                len,
            } => (
                Some(KeptBacking::Shadow {
                    _shadow: shadow.clone(),
                }),
                Some(SlotRefresh::Shadow {
                    shadow,
                    address,
                    len,
                }),
                None,
            ),
        };
        let old = {
            let mut slots = self.kept.lock().unwrap_or_else(|e| e.into_inner());
            slots
                .get_mut(index as usize)
                .and_then(|slot| std::mem::replace(slot, resource))
        };
        if let Some(old) = old {
            // A running submission may still read the old resource through
            // this slot.
            crate::host::retire(old);
        }
        {
            let mut map = self.srv_mips.lock().unwrap_or_else(|e| e.into_inner());
            match host_mips {
                Some(entry) => {
                    map.insert(index, entry);
                }
                None if !map.is_empty() => {
                    map.remove(&index);
                }
                None => {}
            }
        }
        if refresh.is_some() || self.refresh_count.load(Ordering::Acquire) != 0 {
            let mut map = self.refreshes.lock().unwrap_or_else(|e| e.into_inner());
            match refresh {
                Some(entry) => {
                    map.insert(index, entry);
                }
                None => {
                    map.remove(&index);
                }
            }
            self.refresh_count.store(map.len(), Ordering::Release);
        }
    }

    /// If slot `index` has a registered refresh entry (aliased shadow or a
    /// view's own copy), schedules a submission-time refresh on `list` and
    /// returns `true`.
    pub(crate) fn refresh_slot(&self, index: u32, list: &HostList) -> bool {
        if self.refresh_count.load(Ordering::Acquire) == 0 {
            return false;
        }
        let refresh = {
            let map = self.refreshes.lock().unwrap_or_else(|e| e.into_inner());
            if map.is_empty() {
                return false;
            }
            map.get(&index).cloned()
        };
        match refresh {
            Some(SlotRefresh::Shadow {
                shadow,
                address,
                len,
            }) => {
                shadow.refresh(list, address, len);
                true
            }
            None => false,
        }
    }

    /// Makes slot `index` (a dynamic one) a constant buffer view of host
    /// memory (the list's own, which it holds).
    pub(crate) fn constant_at(&self, index: u32, gpu: u64, size: u32) {
        self.keep(index, Kept::Empty);
        let Some(host) = host() else { return };
        let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
            BufferLocation: gpu,
            SizeInBytes: size.next_multiple_of(256),
        };
        // SAFETY: a slot of our heap; the memory is the caller's.
        unsafe {
            host.device
                .CreateConstantBufferView(Some(&desc), self.cpu(index))
        };
    }

    /// Makes slot `index` (a dynamic one) a raw buffer view of `len` bytes
    /// of `shadow` from byte `offset` (a multiple of 16): a root SRV with a
    /// size, whose reads past it return zeros instead of reaching whatever
    /// the host placed after the buffer.
    pub(crate) fn raw_at(&self, index: u32, shadow: &Arc<Shadow>, offset: u64, len: u64) {
        self.write_raw_at(
            index,
            &shadow.buffer,
            Kept::Shadow(Arc::clone(shadow)),
            offset,
            len,
        );
    }

    /// [`Self::raw_at`] over a buffer of our own that no shadow backs.
    pub(crate) fn raw_resource_at(
        &self,
        index: u32,
        buffer: &ID3D12Resource,
        offset: u64,
        len: u64,
    ) {
        self.write_raw_at(index, buffer, Kept::Resource(buffer.clone()), offset, len);
    }

    fn write_raw_at(&self, index: u32, buffer: &ID3D12Resource, kept: Kept, offset: u64, len: u64) {
        let Some(elements) = raw_element_count(len) else {
            self.null_raw_at(index);
            return;
        };
        let Some(host) = host() else { return };
        let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: offset / 4,
                    NumElements: elements.get(),
                    StructureByteStride: 0,
                    Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                },
            },
        };
        // SAFETY: a slot of our heap and a live buffer covering the range.
        unsafe {
            host.device
                .CreateShaderResourceView(buffer, Some(&desc), self.cpu(index))
        };
        self.keep(index, kept);
    }

    /// Makes slot `index` a raw buffer view that reads zeros.
    pub(crate) fn null_raw_at(&self, index: u32) {
        null_buffer_view(self.cpu(index));
        self.keep(index, Kept::Empty);
    }

    /// Makes slot `index` (a dynamic one) a raw buffer view with unordered
    /// access of `len` bytes of `shadow` from byte `offset` (a multiple of
    /// 16): a root UAV with a size, whose writes past it are dropped
    /// instead of reaching whatever the host placed after the buffer.
    pub(crate) fn raw_shadow_uav_at(
        &self,
        index: u32,
        shadow: &Arc<Shadow>,
        offset: u64,
        len: u64,
    ) {
        self.write_raw_uav_at(
            index,
            &shadow.buffer,
            Kept::Shadow(Arc::clone(shadow)),
            offset,
            len,
        );
    }

    pub(crate) fn raw_uav_at(&self, index: u32, buffer: &ID3D12Resource, offset: u64, len: u64) {
        self.write_raw_uav_at(index, buffer, Kept::Resource(buffer.clone()), offset, len);
    }

    fn write_raw_uav_at(
        &self,
        index: u32,
        buffer: &ID3D12Resource,
        kept: Kept,
        offset: u64,
        len: u64,
    ) {
        let Some(host) = host() else { return };
        let Some(elements) = raw_element_count(len) else {
            self.null_raw_uav_at(index);
            return;
        };
        let desc = raw_uav_desc(offset / 4, elements.get());
        // SAFETY: a slot of our heap and a live buffer covering the range.
        unsafe {
            host.device.CreateUnorderedAccessView(
                buffer,
                None::<&ID3D12Resource>,
                Some(&desc),
                self.cpu(index),
            )
        };
        self.keep(index, kept);
    }

    /// Makes slot `index` a raw buffer view with unordered access of no
    /// buffer: reads return zeros and writes are dropped.
    pub(crate) fn null_raw_uav_at(&self, index: u32) {
        let Some(host) = host() else { return };
        let desc = raw_uav_desc(0, 1);
        // SAFETY: a slot of our heap; a null resource is allowed with a desc.
        unsafe {
            host.device.CreateUnorderedAccessView(
                None::<&ID3D12Resource>,
                None::<&ID3D12Resource>,
                Some(&desc),
                self.cpu(index),
            )
        };
        self.keep(index, Kept::Empty);
    }

    /// The shader-visible handle of slot `index`.
    pub(crate) fn gpu_handle(&self, index: u32) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: self.gpu + u64::from(index) * u64::from(self.step),
        }
    }

    fn cpu(&self, index: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.cpu + index as usize * self.step as usize,
        }
    }

    /// Makes host slot `index` from the record the title has at `handle`
    /// if it changed since, and returns the view recorded there. Buffer
    /// views are made only for draws (`for_draw`), which also copy their
    /// data: a host copy of a buffer is made on first use.
    pub(crate) fn mirror(&self, index: u32, handle: u64, for_draw: bool) -> Option<View> {
        if index >= self.count {
            // Diagnostic (NVIDIA heap cap): the highest title slot past the
            // host heap seen so far.
            static HIGHEST: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if HIGHEST.fetch_max(index, std::sync::atomic::Ordering::Relaxed) < index {
                tracing::warn!(
                    index,
                    count = self.count,
                    for_draw,
                    "title slot past the host heap"
                );
            }
            return None;
        }
        // SAFETY: the handle is a descriptor slot in title memory.
        let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
        // Generations start at 1, so a made slot's key is never 0.
        let key = u64::from(generation) << 32 | u64::from(i);
        let _slot = self.stripes[index as usize % SLOT_STRIPES]
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let made = &self.made[index as usize];
        let view = (tag == VIEW_TAG).then(|| view_at(i, generation)).flatten();
        let Some(view) = view else {
            // Not a view of ours (never written, or of a resource the
            // title released): nothing to read.
            if made.load(Ordering::Acquire) != 0 {
                made.store(0, Ordering::Release);
                self.make_null(index);
                self.track_slot(index, None);
                self.track_buffer(index, None);
            }
            return None;
        };
        // A streamed texture's host SRV may have been narrowed to the levels
        // that had arrived; remake when more (or all) of them are there.
        let widen = for_draw && !self.sampler && made.load(Ordering::Acquire) == key && {
            let map = self.srv_mips.lock().unwrap_or_else(|e| e.into_inner());
            match map.get(&index) {
                Some(&(title, host)) => residency_widened(&view, title, host),
                None => false,
            }
        };
        if made.load(Ordering::Acquire) != key || widen {
            let later = !for_draw && view.is_buffer();
            made.store(if later { 0 } else { key }, Ordering::Release);
            let kept = if later {
                None
            } else {
                make(&view, self.cpu(index))
            };
            let made_ok = kept.is_some();
            let new_entry = match &kept {
                Some(Kept::Texture { title, is_uav, .. }) => {
                    Some((*title, *is_uav, placed_first_mip(&view)))
                }
                _ => None,
            };
            // Ordinary buffer / constant copies are not in `refreshes` (only
            // unaligned and aliased structured slots are); record them so
            // bindless draws can refresh the ranges they index.
            let buf = match &kept {
                Some(Kept::Shadow(_)) => buffer_refresh_range(&view),
                _ => None,
            };
            if !made_ok && !self.sampler {
                null_for(&view, self.cpu(index));
            }
            self.keep(index, kept.unwrap_or(Kept::Empty));
            self.track_slot(index, new_entry);
            self.track_buffer(index, buf);
            // Close the race with `descriptor::forget` running between the initial
            // `view_at` lookup and `track_slot` before this heap's publication.
            if new_entry.is_some() && view_at(i, generation).is_none() {
                made.store(0, Ordering::Release);
                self.make_null(index);
                self.track_slot(index, None);
                self.track_buffer(index, None);
                return None;
            }
        }
        Some(view)
    }

    /// Scans title descriptor slots `[first, end)` and mirrors any slot whose
    /// on-heap `(tag, view_index, generation)` differs from what the host slot
    /// was last made from, catching descriptors the title copied directly in
    /// heap memory without calling a descriptor method.
    pub(crate) fn sync_slots(
        &self,
        start: u64,
        step: u64,
        first: u32,
        end: u32,
        for_draw: bool,
    ) -> usize {
        let end = end.min(self.count);
        if first >= end || step == 0 {
            return 0;
        }
        let mut changed = Vec::new();
        {
            let made = &self.made;
            for index in first..end {
                let handle = start + u64::from(index) * step;
                // SAFETY: `[first, end)` is bounded by `self.count` (which does
                // not exceed the title heap's descriptor count) and each
                // descriptor slot is at least 16 bytes.
                let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
                let expected = if tag == VIEW_TAG {
                    u64::from(generation) << 32 | u64::from(i)
                } else {
                    0
                };
                if made[index as usize].load(Ordering::Acquire) != expected {
                    changed.push(index);
                }
            }
        }
        let mut mirrored = 0;
        for index in changed {
            let handle = start + u64::from(index) * step;
            self.mirror(index, handle, for_draw);
            mirrored += 1;
        }
        mirrored
    }

    /// Remakes host SRVs that were narrowed to a streamed mip suffix once more
    /// (or all) levels have arrived, even when the title descriptor itself did
    /// not change — bindless ResourceHeap draws never walk tables, and
    /// write-watch sync only remirrors dirty heap pages (#616).
    pub(crate) fn remirror_widened_srvs(&self, start: u64, step: u64, for_draw: bool) -> usize {
        if !for_draw || self.sampler || step == 0 {
            return 0;
        }
        let candidates: Vec<(u32, SrvMips)> = {
            let map = self.srv_mips.lock().unwrap_or_else(|e| e.into_inner());
            map.iter()
                .filter(|&(&index, _)| index < self.count)
                .map(|(&index, &entry)| (index, entry))
                .collect()
        };
        let mut mirrored = 0;
        for (index, (title, host_mips)) in candidates {
            let handle = start + u64::from(index) * step;
            // SAFETY: slot index is below `self.count`; each descriptor is ≥16 bytes.
            let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
            let Some(view) = (tag == VIEW_TAG).then(|| view_at(i, generation)).flatten() else {
                continue;
            };
            if residency_widened(&view, title, host_mips) {
                self.mirror(index, handle, for_draw);
                mirrored += 1;
            }
        }
        mirrored
    }

    /// Updates the leaf-only live texture tracker for slot `index`.
    fn track_slot(&self, index: u32, new_entry: Option<(usize, bool, u32)>) {
        if self.sampler {
            return;
        }
        let mut tracker = self.tex_tracker.lock().unwrap_or_else(|e| e.into_inner());
        let Some(slot) = tracker.slot_res.get_mut(index as usize) else {
            return;
        };
        let old_entry = *slot;
        if old_entry == new_entry {
            return;
        }
        *slot = new_entry;
        if let Some((res, is_uav, _)) = old_entry {
            tracker.remove_entry(index, res, is_uav);
        }
        if let Some((res, is_uav, _)) = new_entry {
            tracker.add_entry(index, res, is_uav);
        }
        tracker.generation += 1;
        self.tex_gen.store(tracker.generation, Ordering::Release);
        if let Some((res, _, _)) = new_entry {
            tracker.record_change(res);
        }
    }

    /// Updates the live buffer-range tracker for title slot `index`.
    fn track_buffer(&self, index: u32, range: Option<BufRange>) {
        if self.sampler || index >= self.count {
            return;
        }
        let mut tracker = self.buf_tracker.lock().unwrap_or_else(|e| e.into_inner());
        match range {
            Some(range) => {
                tracker.live.insert(index, range);
            }
            None => {
                tracker.live.remove(&index);
            }
        }
    }

    /// Brings title-memory buffer ranges viewed by this heap's title slots up
    /// to date for a draw or dispatch that indexes `ResourceDescriptorHeap`.
    ///
    /// Covers ordinary buffer / constant copies tracked in [`BufTracker`], and
    /// unaligned / aliased structured slots registered in [`Self::refreshes`].
    /// Unordered heap views refresh into unordered-access state so the
    /// dispatch can write them; copy-back of heap-only UAV ranges is still
    /// skipped (a full-heap snapshot is too expensive; table UAVs still go
    /// through [`crate::draw`]'s `readable` path).
    pub(crate) fn refresh_bindless_buffers(&self, list: &HostList) {
        if self.sampler {
            return;
        }
        let special: Vec<u32> = {
            let map = self.refreshes.lock().unwrap_or_else(|e| e.into_inner());
            map.keys().copied().filter(|&i| i < self.count).collect()
        };
        for index in special {
            let _ = self.refresh_slot(index, list);
        }
        let live: Vec<BufRange> = {
            let tracker = self.buf_tracker.lock().unwrap_or_else(|e| e.into_inner());
            tracker.live.values().cloned().collect()
        };
        for (shadow, address, len, is_uav) in live {
            let access = if is_uav {
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS
            } else {
                D3D12_RESOURCE_STATE_GENERIC_READ
            };
            shadow.refresh_as(list, address, len, access);
        }
    }

    /// Live buffer ranges currently tracked for bindless refresh (tests).
    #[cfg(test)]
    pub(crate) fn live_buffer_count(&self) -> usize {
        self.buf_tracker
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .live
            .len()
    }

    /// Removes all tracked references to `resource` when the title releases it,
    /// clearing only the exact slots that currently held `resource`.
    pub(crate) fn forget_resource(&self, resource: usize) {
        if self.sampler {
            return;
        }
        let mut tracker = self.tex_tracker.lock().unwrap_or_else(|e| e.into_inner());
        let mut changed = false;
        for is_uav in [false, true] {
            if let Some(slots) = tracker.res_slots.remove(&(resource, is_uav)) {
                for slot in slots {
                    if let Some(entry) = tracker.slot_res.get_mut(slot as usize)
                        && entry.is_some_and(|(r, u, _)| (r, u) == (resource, is_uav))
                    {
                        *entry = None;
                    }
                }
                let counts = if is_uav {
                    &mut tracker.live_uavs
                } else {
                    &mut tracker.live_srvs
                };
                counts.remove(&resource);
                changed = true;
            }
        }
        if changed {
            tracker.generation += 1;
            self.tex_gen.store(tracker.generation, Ordering::Release);
        }
    }

    /// Current generation of tracked live textures in this heap.
    pub(crate) fn tex_gen(&self) -> u64 {
        self.tex_gen.load(Ordering::Acquire)
    }

    /// Snapshot of `(tex_gen, live_srvs, live_uavs)` in `O(active)` time,
    /// pinning each live title `Resource` with [`crate::object::Held`] while
    /// `tex_tracker` is locked so concurrent releases cannot free it mid-pass.
    pub(crate) fn live_textures(&self, srvs: bool, uavs: bool) -> LiveTextures {
        let tracker = self.tex_tracker.lock().unwrap_or_else(|e| e.into_inner());
        let pin = |r: &usize| {
            // SAFETY: while `tex_tracker` is held, any tracked `Resource` has
            // not finished `Resource::drop` (which calls `forget_resource`
            // before deallocating), so its `refs` atomic is still allocated.
            unsafe { crate::object::Held::try_new(*r as crate::object::This) }
        };
        let srv_list = if srvs {
            tracker
                .live_srvs
                .keys()
                .filter_map(|r| Some((pin(r)?, tracker.first_mip(*r))))
                .collect()
        } else {
            Vec::new()
        };
        let uav_list = if uavs {
            tracker.live_uavs.keys().filter_map(pin).collect()
        } else {
            Vec::new()
        };
        (tracker.generation, srv_list, uav_list)
    }

    /// Pins the subset of resources that changed in this heap since
    /// `since_gen` plus `extra_candidates` (e.g., list-dirtied textures and
    /// previously skipped render targets), returning `None` if the bounded
    /// change ring wrapped past `since_gen` so the caller falls back to a full
    /// `live_textures` walk.
    pub(crate) fn pin_live_subset_since(
        &self,
        since_gen: u64,
        extra_candidates: &[usize],
        srvs: bool,
        uavs: bool,
    ) -> Option<LiveTextures> {
        let tracker = self.tex_tracker.lock().unwrap_or_else(|e| e.into_inner());
        if since_gen < tracker.oldest_gen {
            return None;
        }
        let mut srv_list = Vec::new();
        let mut uav_list = Vec::new();
        let mut seen = HashSet::with_capacity(extra_candidates.len());
        let mut visit = |r: usize| {
            if !seen.insert(r) {
                return;
            }
            let is_srv = srvs && tracker.live_srvs.contains_key(&r);
            let is_uav = uavs && tracker.live_uavs.contains_key(&r);
            if !is_srv && !is_uav {
                return;
            }
            // SAFETY: while `tex_tracker` is held, any tracked `Resource` has
            // not finished `Resource::drop` (which calls `forget_resource`
            // before deallocating), so its `refs` atomic is still allocated.
            let Some(held) = (unsafe { crate::object::Held::try_new(r as crate::object::This) })
            else {
                return;
            };
            if is_srv && is_uav {
                if let Some(second) =
                    // SAFETY: `held` already pins the resource alive here.
                    unsafe { crate::object::Held::try_new(r as crate::object::This) }
                {
                    uav_list.push(second);
                }
                srv_list.push((held, tracker.first_mip(r)));
            } else if is_srv {
                srv_list.push((held, tracker.first_mip(r)));
            } else {
                uav_list.push(held);
            }
        };
        for &r in extra_candidates {
            visit(r);
        }
        if since_gen < tracker.generation {
            for &(gen_id, r) in &tracker.changes {
                if gen_id > since_gen {
                    visit(r);
                }
            }
        }
        Some((tracker.generation, srv_list, uav_list))
    }

    /// Raw tracked addresses for synthetic unit tests that exercise
    /// `TexTracker` ABA safety without real COM objects.
    #[cfg(test)]
    fn live_texture_addrs(&self, srvs: bool, uavs: bool) -> (u64, Vec<usize>, Vec<usize>) {
        let tracker = self.tex_tracker.lock().unwrap_or_else(|e| e.into_inner());
        let srv_list = if srvs {
            tracker.live_srvs.keys().copied().collect()
        } else {
            Vec::new()
        };
        let uav_list = if uavs {
            tracker.live_uavs.keys().copied().collect()
        } else {
            Vec::new()
        };
        (tracker.generation, srv_list, uav_list)
    }

    /// The view host slot `index` was made from, in a few words, with the
    /// title and host addresses of a buffer view (diagnostics). "null" for
    /// a slot never made or left null.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn describe_slot(&self, index: u32) -> Option<String> {
        let key = self.made.get(index as usize)?.load(Ordering::Acquire);
        if key == 0 {
            return Some("null".into());
        }
        let view = view_at(key as u32, (key >> 32) as u32)?;
        let mut text: String = format!("{view:?}").chars().take(120).collect();
        if let Some((address, len)) = buffer_range(&view) {
            let host = crate::shadow::find(address, len.max(1)).map(|s| s.gpu(address));
            text.push_str(&format!(" title {address:#x}+{len:#x} host {host:#x?}"));
        }
        Some(text)
    }
}

/// Logs `what` the first time it happens.
macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// Narrows a shader resource view of `texture` to the mips of `resource`
/// whose memory is there: `MostDetailedMip` up to the most detailed resident
/// one, `MipLevels` to what is left. Returns the mips the host view names,
/// `None` if the view is left as the title asked (fully mapped, host-written,
/// not a 2D view, or nothing known yet).
fn narrow_to_resident(
    resource: crate::object::This,
    texture: &ID3D12Resource,
    desc: &mut D3D12_SHADER_RESOURCE_VIEW_DESC,
) -> Option<(u32, u32)> {
    // A copy/resolve filled the host mip chain; title-memory residency is
    // stale and must not clamp the view (#616).
    if crate::texture::is_host_written(resource) {
        return None;
    }
    // Prefer the window the last staging refresh noted; fall back to a live
    // mapped-prefix query so the first view before any upload is still clamped.
    let resident =
        crate::texture::resident(resource).or_else(|| crate::texture::live_resident(resource))?;
    if resident.1 == 0 {
        return None;
    }
    // SAFETY: a live resource.
    let mips = u32::from(unsafe { texture.GetDesc() }.MipLevels);
    let asked = match desc.ViewDimension {
        D3D12_SRV_DIMENSION_TEXTURE2D => {
            // SAFETY: the dimension selects the Texture2D union member.
            let t = unsafe { desc.Anonymous.Texture2D };
            (t.MostDetailedMip, t.MipLevels)
        }
        D3D12_SRV_DIMENSION_TEXTURE2DARRAY => {
            // SAFETY: the dimension selects the Texture2DArray union member.
            let t = unsafe { desc.Anonymous.Texture2DArray };
            (t.MostDetailedMip, t.MipLevels)
        }
        _ => return None,
    };
    let (first, count) = crate::texture::narrowed(asked, resident, mips);
    let asked_levels = if asked.1 == 0 || asked.1 == u32::MAX {
        mips.saturating_sub(asked.0)
    } else {
        asked.1
    };
    if (first, count) == (asked.0, asked_levels) {
        return None;
    }
    match desc.ViewDimension {
        D3D12_SRV_DIMENSION_TEXTURE2D => {
            // SAFETY: as above.
            let t = unsafe { &mut desc.Anonymous.Texture2D };
            *t = D3D12_TEX2D_SRV {
                MostDetailedMip: first,
                MipLevels: count,
                PlaneSlice: t.PlaneSlice,
                ResourceMinLODClamp: t.ResourceMinLODClamp,
            };
        }
        _ => {
            // SAFETY: as above.
            let t = unsafe { &mut desc.Anonymous.Texture2DArray };
            *t = D3D12_TEX2D_ARRAY_SRV {
                MostDetailedMip: first,
                MipLevels: count,
                FirstArraySlice: t.FirstArraySlice,
                ArraySize: t.ArraySize,
                PlaneSlice: t.PlaneSlice,
                ResourceMinLODClamp: t.ResourceMinLODClamp,
            };
        }
    }
    Some((first, count))
}

/// Whether a host SRV made with `host_mips` for title view `view` should be
/// remade because more (or all) streamed levels have arrived.
fn residency_widened(view: &View, title: usize, host_mips: (u32, u32)) -> bool {
    let Some(texture) = crate::resource::host_resource(title as crate::object::This) else {
        return false;
    };
    // SAFETY: a live resource when this slot still holds it.
    let rd = unsafe { texture.GetDesc() };
    let mips = u32::from(rd.MipLevels);
    let asked = match view {
        View::Shader(_, Some(w)) => match asked_srv_mips(&srv_desc(w), mips) {
            Some(a) => a,
            None => return false,
        },
        View::Placed(false, _, desc) => {
            let words: [u32; 10] = match desc[..10].try_into() {
                Ok(w) => w,
                Err(_) => return false,
            };
            match asked_srv_mips(&srv_desc(&words), mips) {
                Some(a) => a,
                None => return false,
            }
        }
        // Typeless default SRVs synthesize a concrete desc at make time; rebuild
        // that same asked window so a later widen can remake them (#616).
        View::Shader(_, None) => {
            let own = rd.Format.0 as u32;
            if !is_typeless(own) {
                return false;
            }
            let Some(fmt) = concrete_format_for(title as crate::object::This, own, false) else {
                return false;
            };
            let Some(desc) = default_srv_desc(&rd, fmt) else {
                return false;
            };
            match asked_srv_mips(&desc, mips) {
                Some(a) => a,
                None => return false,
            }
        }
        _ => return false,
    };
    let want = if crate::texture::is_host_written(title as crate::object::This) {
        // Host copy owns the full chain; restore the title's asked range.
        if host_mips != asked {
            asked
        } else {
            return false;
        }
    } else {
        match crate::texture::resident(title as crate::object::This) {
            Some(window) if window.1 > 0 => crate::texture::narrowed(asked, window, mips),
            // Fully mapped (or no longer tracked as partial): the title's range.
            _ if host_mips != asked => asked,
            _ => return false,
        }
    };
    want != host_mips
}

/// `(MostDetailedMip, MipLevels)` a 2D / 2D-array SRV asks for, with `0` /
/// [`u32::MAX`] levels expanded against the resource mip count.
fn asked_srv_mips(desc: &D3D12_SHADER_RESOURCE_VIEW_DESC, mips: u32) -> Option<(u32, u32)> {
    match desc.ViewDimension {
        D3D12_SRV_DIMENSION_TEXTURE2D => {
            // SAFETY: Texture2D member.
            let t = unsafe { desc.Anonymous.Texture2D };
            let levels = if t.MipLevels == 0 || t.MipLevels == u32::MAX {
                mips.saturating_sub(t.MostDetailedMip)
            } else {
                t.MipLevels
            };
            Some((t.MostDetailedMip, levels))
        }
        D3D12_SRV_DIMENSION_TEXTURE2DARRAY => {
            // SAFETY: Texture2DArray member.
            let t = unsafe { desc.Anonymous.Texture2DArray };
            let levels = if t.MipLevels == 0 || t.MipLevels == u32::MAX {
                mips.saturating_sub(t.MostDetailedMip)
            } else {
                t.MipLevels
            };
            Some((t.MostDetailedMip, levels))
        }
        _ => None,
    }
}

/// A view description from its words, with the format as the host has it.
fn srv_desc(words: &[u32; 10]) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_SHADER_RESOURCE_VIEW_DESC>()
            .read_unaligned()
    }
}

fn uav_desc(words: &[u32; 10]) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_UNORDERED_ACCESS_VIEW_DESC>()
            .read_unaligned()
    }
}

/// The stride of a matching structured or raw buffer view. Structured
/// views report their element size; a raw view reports 0 so the shader
/// translator can treat an indexed access as a flat byte buffer (#462).
pub(crate) fn structured_stride(view: &View, uav: bool) -> Option<u32> {
    let words = match (view, uav) {
        (View::Shader(_, Some(words)), false) | (View::Unordered(_, _, Some(words)), true) => words,
        (View::Placed(unordered, resource, words), _) if *unordered == uav && resource[0] == 1 => {
            words[..10].try_into().ok()?
        }
        _ => return None,
    };
    let stride = if uav {
        let desc = uav_desc(words);
        if desc.ViewDimension != D3D12_UAV_DIMENSION_BUFFER {
            return None;
        }
        // SAFETY: The view dimension selects the buffer member.
        let buffer = unsafe { desc.Anonymous.Buffer };
        match (desc.Format, buffer.Flags, buffer.StructureByteStride) {
            (DXGI_FORMAT_UNKNOWN, D3D12_BUFFER_UAV_FLAG_RAW, _)
            | (DXGI_FORMAT_R32_TYPELESS, D3D12_BUFFER_UAV_FLAG_RAW, 0) => {
                return Some(0);
            }
            (DXGI_FORMAT_UNKNOWN, D3D12_BUFFER_UAV_FLAG_NONE, stride) => stride,
            _ => return None,
        }
    } else {
        let desc = srv_desc(words);
        if desc.ViewDimension != D3D12_SRV_DIMENSION_BUFFER {
            return None;
        }
        // SAFETY: The view dimension selects the buffer member.
        let buffer = unsafe { desc.Anonymous.Buffer };
        match (desc.Format, buffer.Flags, buffer.StructureByteStride) {
            (DXGI_FORMAT_UNKNOWN, D3D12_BUFFER_SRV_FLAG_RAW, _)
            | (DXGI_FORMAT_R32_TYPELESS, D3D12_BUFFER_SRV_FLAG_RAW, 0) => {
                return Some(0);
            }
            (DXGI_FORMAT_UNKNOWN, D3D12_BUFFER_SRV_FLAG_NONE, stride) => stride,
            _ => return None,
        }
    };
    ((4..=2048).contains(&stride) && stride.is_multiple_of(4)).then_some(stride)
}

/// A buffer view of title memory at `address` on its host copy: the first
/// element moves by the copy's offset.
/// Bytes per element of a buffer view.
fn element_size(desc: &D3D12_SHADER_RESOURCE_VIEW_DESC) -> u64 {
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    }
}

/// Host copy range a buffer or constant view reads, for bindless refresh.
///
/// Resolves the shadow the same way descriptor creation does (`find` on the
/// buffer's full base and length), then refreshes only the viewed subrange —
/// so overlapping title buffers do not pick a different innermost shadow than
/// the one the descriptor was created against.
fn buffer_refresh_range(view: &View) -> Option<BufRange> {
    let is_uav = matches!(view, View::Unordered(..) | View::Placed(true, ..));
    let (base, extent) = buffer_host_extent(view)?;
    if extent == 0 {
        return None;
    }
    let shadow = crate::shadow::find(base, extent)?;
    let (address, len) = match *view {
        View::Constant(address, size) => (address, u64::from(size)),
        _ => buffer_range(view)?,
    };
    if len == 0 {
        return None;
    }
    let len = len.min(shadow.left(address));
    (len > 0).then_some((shadow, address, len, is_uav))
}

/// The full title buffer a view was created against: base address and length
/// passed to [`crate::shadow::find`] when making the host descriptor.
fn buffer_host_extent(view: &View) -> Option<(u64, u64)> {
    match view {
        View::Constant(address, size) => Some((*address, u64::from(*size))),
        View::Shader(resource, Some(_)) | View::Unordered(resource, _, Some(_)) => {
            let (data, len) = crate::resource::buffer_bytes(*resource as crate::object::This)?;
            Some((data as u64, len))
        }
        View::Placed(_, resource, desc) if resource[0] == 1 => {
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            let len = u64::from(resource[4]) | u64::from(resource[5]) << 32;
            Some((address, len))
        }
        _ => None,
    }
}

/// The title memory a buffer shader or unordered access view reads: its
/// address and length.
pub(crate) fn buffer_range(view: &View) -> Option<(u64, u64)> {
    let (base, words): (u64, &[u32; 10]) = match view {
        View::Shader(r, Some(desc)) => (
            crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64,
            desc,
        ),
        View::Placed(false, resource, desc) if resource[0] == 1 => (
            u64::from(desc[10]) | u64::from(desc[11]) << 32,
            desc[..10].try_into().ok()?,
        ),
        View::Unordered(r, _, Some(desc)) => {
            let base = crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64;
            return unordered_range(base, desc);
        }
        View::Placed(true, resource, desc) if resource[0] == 1 => {
            let base = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            return unordered_range(base, desc[..10].try_into().ok()?);
        }
        _ => return None,
    };
    let desc = srv_desc(words);
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    Some((
        base + buffer.FirstElement * element,
        u64::from(buffer.NumElements) * element,
    ))
}

/// [`buffer_range`] of an unordered access view of memory at `base`.
fn unordered_range(base: u64, words: &[u32; 10]) -> Option<(u64, u64)> {
    let desc = uav_desc(words);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32)?.0
    };
    Some((
        base + buffer.FirstElement * element,
        u64::from(buffer.NumElements) * element,
    ))
}

fn raw_uav_desc(first: u64, elements: u32) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_UAV {
                FirstElement: first,
                NumElements: elements,
                StructureByteStride: 0,
                CounterOffsetInBytes: 0,
                Flags: D3D12_BUFFER_UAV_FLAG_RAW,
            },
        },
    }
}

/// Whether a raw buffer view may start at element `first` (4 bytes each):
/// the host wants 16-byte aligned starts, the console only 4, and refuses
/// the call (removing the device) for the rest.
/// shortcut: a view the host refuses reads nothing (a null view); starting
/// it 16 bytes lower would need the shader to add the difference.
fn raw_start_fits(first: u64) -> bool {
    first.is_multiple_of(4)
}

/// The host resource a slot made by [`make`] views, if any.
enum Kept {
    Empty,
    Resource(ID3D12Resource),
    Shadow(Arc<Shadow>),
    /// A host texture SRV or UAV, carrying the title `Resource` address and
    /// access kind so [`HostHeap::mirror`] can track it without re-reading the
    /// title pointer after `make`. `host_mips` is the `(MostDetailedMip,
    /// MipLevels)` the host SRV was made with (`None` for UAVs), so a later
    /// mirror can remake it when more streamed levels arrive.
    Texture {
        resource: ID3D12Resource,
        title: usize,
        is_uav: bool,
        host_mips: Option<(u32, u32)>,
    },
    AliasedShadow {
        shadow: Arc<Shadow>,
        address: u64,
        len: u64,
    },
}

fn buffer_srv_parts(view: &View) -> Option<(u64, u64, D3D12_SHADER_RESOURCE_VIEW_DESC)> {
    const BUFFER: u32 = 1;
    match view {
        View::Shader(resource, Some(desc)) => {
            let (data, len) = crate::resource::buffer_bytes(*resource as crate::object::This)?;
            Some((data as u64, len, srv_desc(desc)))
        }
        View::Placed(false, resource, desc) if resource[0] == BUFFER => {
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            let len = u64::from(resource[4]) | u64::from(resource[5]) << 32;
            Some((address, len, srv_desc(desc[..10].try_into().ok()?)))
        }
        _ => None,
    }
}

pub(crate) struct BufferRead {
    pub(crate) shadow: Arc<crate::shadow::Shadow>,
    pub(crate) address: u64,
    pub(crate) bytes: u64,
    pub(crate) stride: u32,
}

/// A bounded dword-readable window in the backing buffer. Raw views index
/// bytes, even though the descriptor expresses its extent in dwords.
pub(crate) fn selected_buffer(view: &View) -> Option<BufferRead> {
    let (base, size, desc) = buffer_srv_parts(view)?;
    if desc.ViewDimension != D3D12_SRV_DIMENSION_BUFFER {
        return None;
    }
    // SAFETY: the dimension selects the Buffer union member.
    let b = unsafe { desc.Anonymous.Buffer };
    let element = element_size(&desc);
    let stride = if b.Flags == D3D12_BUFFER_SRV_FLAG_RAW {
        1
    } else {
        element.try_into().ok()?
    };
    let start = b.FirstElement.checked_mul(element)?;
    let bytes = u64::from(b.NumElements).checked_mul(element)?;
    if bytes < 4 || start.checked_add(bytes)? > size {
        return None;
    }
    let shadow = crate::shadow::find(base, size)?;
    let address = base.checked_add(start)?;
    let offset = shadow.offset(address);
    // Reserve the last 16-byte block as an out-of-bounds address even for
    // four-dword reads. Include the prefix required by raw SRV alignment.
    if !offset.is_multiple_of(4) || bytes + offset % 16 > u64::from(u32::MAX - 15) {
        return None;
    }
    Some(BufferRead {
        shadow,
        address,
        bytes,
        stride,
    })
}

/// Re-expresses the exact byte start and length of a raw buffer SRV as
/// structured elements of width `stride`, rejecting typed or already-structured
/// views and fractional bounds.
fn structured_alias(
    mut desc: D3D12_SHADER_RESOURCE_VIEW_DESC,
    stride: u32,
) -> Option<D3D12_SHADER_RESOURCE_VIEW_DESC> {
    if desc.ViewDimension != D3D12_SRV_DIMENSION_BUFFER
        || !(4..=2048).contains(&stride)
        || !stride.is_multiple_of(4)
    {
        return None;
    }
    let old_bytes = element_size(&desc);
    // SAFETY: `ViewDimension == D3D12_SRV_DIMENSION_BUFFER` verified above.
    let old = unsafe { desc.Anonymous.Buffer };
    if old.StructureByteStride != 0
        || old.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 == 0
        || old_bytes == 0
        || old.NumElements == 0
    {
        return None;
    }
    let width = u64::from(stride);
    let start = old.FirstElement.checked_mul(old_bytes)?;
    let size = u64::from(old.NumElements).checked_mul(old_bytes)?;
    if start % width != 0 || size % width != 0 {
        return None;
    }
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.Anonymous.Buffer = D3D12_BUFFER_SRV {
        FirstElement: start / width,
        NumElements: u32::try_from(size / width).ok()?,
        StructureByteStride: stride,
        Flags: D3D12_BUFFER_SRV_FLAG_NONE,
    };
    Some(desc)
}

fn buffer_srv(
    address: u64,
    len: u64,
    desc: D3D12_SHADER_RESOURCE_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    buffer_srv_inner(address, len, desc, at, false)
}

fn buffer_srv_inner(
    address: u64,
    len: u64,
    mut desc: D3D12_SHADER_RESOURCE_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
    track_shadow: bool,
) -> Option<Kept> {
    let shadow = crate::shadow::find(address, len)?;
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    let offset = shadow.offset(address);
    let host = host()?;
    if element == 0 {
        once!(
            offset,
            element,
            "buffer view not on an element of its host copy"
        );
        return None;
    }
    // A view of no elements is an invalid call on the host (it removes the
    // device); the title makes such views for buffers it has not sized yet.
    // The caller makes a null view instead.
    if buffer.NumElements == 0 {
        once!("a buffer view of no elements is left null");
        return None;
    }
    let view_offset = buffer.FirstElement.checked_mul(element)?;
    let bytes = u64::from(buffer.NumElements).checked_mul(element)?;
    if track_shadow && view_offset.checked_add(bytes)? > len {
        return None;
    }
    if offset % element != 0 {
        // The title's placed buffer starts at an address that is not a
        // multiple of `element` inside our larger host shadow buffer (e.g. a
        // non-power-of-two structured element stride inside a ring buffer).
        // Give the view its own host copy that starts at the view so elements
        // land where the title put them. The slot holds it; refresh copies
        // only the pages the title wrote.
        //
        // shortcut: prepared from title memory, so it does not see what a
        // shader of the same submission wrote in that range before the read.
        // Getting that right needs host buffer states across the submission.
        let start = address.checked_add(view_offset)?;
        if bytes == 0 {
            once!(
                offset,
                element,
                "buffer view not on an element of its host copy"
            );
            return None;
        }
        let copy = crate::shadow::aligned(start, bytes)?;
        buffer.FirstElement = 0;
        // SAFETY: a live buffer and a slot of our heap.
        unsafe {
            host.device
                .CreateShaderResourceView(&copy.buffer, Some(&desc), at)
        };
        return Some(Kept::AliasedShadow {
            shadow: copy,
            address: start,
            len: bytes,
        });
    }
    buffer.FirstElement += offset / element;
    if buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0 && !raw_start_fits(buffer.FirstElement) {
        let first = buffer.FirstElement;
        once!(first, "raw buffer view not on 16 bytes of its host copy");
        return None;
    }
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device
            .CreateShaderResourceView(&shadow.buffer, Some(&desc), at)
    };
    if track_shadow {
        Some(Kept::AliasedShadow {
            shadow,
            address: address.checked_add(view_offset)?,
            len: bytes,
        })
    } else {
        Some(Kept::Shadow(shadow))
    }
}

/// An unordered access view of title buffer memory at `address` (`len`
/// bytes) on its host copy, as [`buffer_srv`]; false if the copy cannot be
/// written by shaders (upload memory).
/// shortcut: what shaders write stays in the host copy; title memory (and
/// so the CPU, and copies run on the CPU) does not see it.
fn buffer_uav(
    address: u64,
    len: u64,
    mut desc: D3D12_UNORDERED_ACCESS_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    let shadow = crate::shadow::find(address, len)?;
    // SAFETY: a live buffer.
    let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
    if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
        once!("unordered access to a buffer whose host copy shaders cannot write");
        return None;
    }
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    };
    let offset = shadow.offset(address);
    if element == 0 || offset % element != 0 {
        once!(
            offset,
            element,
            "unordered buffer view not on an element of its host copy"
        );
        return None;
    }
    // As in `buffer_srv`.
    if buffer.NumElements == 0 {
        once!("an unordered buffer view of no elements is left null");
        return None;
    }
    buffer.FirstElement += offset / element;
    if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 && !raw_start_fits(buffer.FirstElement) {
        let first = buffer.FirstElement;
        once!(
            first,
            "raw unordered buffer view not on 16 bytes of its host copy"
        );
        return None;
    }
    buffer.CounterOffsetInBytes = 0;
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device.CreateUnorderedAccessView(
            &shadow.buffer,
            None::<&ID3D12Resource>,
            Some(&desc),
            at,
        )
    };
    Some(Kept::Shadow(shadow))
}

/// Whether the host lets views of a typed resource name another format of
/// its family (`CastingFullyTypedFormatSupported`; never SNORM for UNORM,
/// nor FLOAT for another type), which [`fits`] uses for sRGB and linear
/// twins only.
fn casts_typed() -> bool {
    static CASTS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CASTS.get_or_init(|| {
        let Some(host) = host() else { return false };
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS3::default();
        // SAFETY: the query writes its own structure of the given size.
        let asked = unsafe {
            host.device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS3,
                (&raw mut options).cast(),
                size_of_val(&options) as u32,
            )
        };
        let casts = asked.is_ok() && options.CastingFullyTypedFormatSupported.as_bool();
        if !casts {
            tracing::warn!(
                "the host does not cast typed formats: sRGB views of linear textures (and the reverse) read in the texture's own format"
            );
        }
        casts
    })
}

/// Whether `format` is a typeless DXGI format (the first of a multi-format
/// family).
fn is_typeless(format: u32) -> bool {
    use crate::resource::family;
    family(format) == format && family(format + 1) == format
}

/// Whether the active host device supports typed UAV access (`UAV_TYPED_LOAD`
/// or `UAV_TYPED_STORE`, plus `TYPED_UNORDERED_ACCESS_VIEW`) for optional
/// typed-UAV formats such as `B8G8R8A8_UNORM` / `B8G8R8X8_UNORM`.
fn supports_typed_uav(format: DXGI_FORMAT) -> bool {
    static CACHE: std::sync::LazyLock<Mutex<HashMap<i32, bool>>> =
        std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));
    if let Some(&supported) = CACHE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&format.0)
    {
        return supported;
    }
    let supported = host().is_some_and(|h| {
        let mut data = D3D12_FEATURE_DATA_FORMAT_SUPPORT {
            Format: format,
            ..Default::default()
        };
        // SAFETY: query writes its own `D3D12_FEATURE_DATA_FORMAT_SUPPORT` struct.
        let ok = unsafe {
            h.device
                .CheckFeatureSupport(
                    D3D12_FEATURE_FORMAT_SUPPORT,
                    (&raw mut data).cast(),
                    size_of_val(&data) as u32,
                )
                .is_ok()
        };
        ok && (data.Support1.0 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW.0 != 0)
            && (data.Support2.0
                & (D3D12_FORMAT_SUPPORT2_UAV_TYPED_LOAD.0
                    | D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE.0)
                != 0)
    });
    CACHE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(format.0, supported);
    supported
}

/// Default concrete format for a typeless texture when a shader or unordered
/// access view leaves the description absent or `DXGI_FORMAT_UNKNOWN`. Depth
/// multi-plane, block-compressed, and 96-bit `R32G32B32` formats only allow
/// shader views (`None` for UAVs); every typed UAV format is gated on the host
/// device's cached `D3D12_FEATURE_FORMAT_SUPPORT` query.
///
/// Production null/UNKNOWN resolution goes through [`concrete_format_for`]
/// (and `null_view_format`); this helper is the direct typeless-family table
/// used by the unit tests below.
#[cfg(test)]
fn default_concrete_format(typeless: u32, unordered: bool) -> Option<u32> {
    let concrete = crate::resource::typeless_default_format(typeless, unordered)?;
    if unordered && !supports_typed_uav(DXGI_FORMAT(concrete as i32)) {
        return None;
    }
    Some(concrete)
}

/// Resolves the concrete view format for a typeless host texture when the title
/// passed a null description or `DXGI_FORMAT_UNKNOWN`: uses [`view_format`] for
/// textures promoted from a typed console format (sRGB kept on SRVs, mapped to
/// linear on UAVs; depth mapped to an SRV plane format), else
/// [`default_concrete_format`]. (#438 VIC-BT-004)
fn concrete_format_for(
    resource: crate::object::This,
    own: u32,
    unordered: bool,
) -> Option<DXGI_FORMAT> {
    let title = crate::resource::view_format(resource);
    let concrete = crate::resource::null_view_format(own, title, unordered)?;
    if unordered && !supports_typed_uav(DXGI_FORMAT(concrete as i32)) {
        return None;
    }
    Some(DXGI_FORMAT(concrete as i32))
}

/// Rebases a shader view of a title texture onto its host copy, which lacks
/// the title's top `dropped` mip levels (see `resource::dropped_mips`): the
/// level range and the minimum LOD clamp name the title's levels.
fn drop_mips_in_srv(desc: &mut D3D12_SHADER_RESOURCE_VIEW_DESC, dropped: u32) {
    if dropped == 0 {
        return;
    }
    let rebase = |most: &mut u32, levels: &mut u32, clamp: &mut f32| {
        (*most, *levels) = crate::resource::rebase_mip_range(*most, *levels, dropped);
        *clamp = (*clamp - dropped as f32).max(0.0);
    };
    // SAFETY: each arm reads the union member the view dimension selects.
    unsafe {
        match desc.ViewDimension {
            D3D12_SRV_DIMENSION_TEXTURE2D => {
                let v = &mut desc.Anonymous.Texture2D;
                rebase(
                    &mut v.MostDetailedMip,
                    &mut v.MipLevels,
                    &mut v.ResourceMinLODClamp,
                );
            }
            D3D12_SRV_DIMENSION_TEXTURE2DARRAY => {
                let v = &mut desc.Anonymous.Texture2DArray;
                rebase(
                    &mut v.MostDetailedMip,
                    &mut v.MipLevels,
                    &mut v.ResourceMinLODClamp,
                );
            }
            D3D12_SRV_DIMENSION_TEXTURECUBE => {
                let v = &mut desc.Anonymous.TextureCube;
                rebase(
                    &mut v.MostDetailedMip,
                    &mut v.MipLevels,
                    &mut v.ResourceMinLODClamp,
                );
            }
            D3D12_SRV_DIMENSION_TEXTURECUBEARRAY => {
                let v = &mut desc.Anonymous.TextureCubeArray;
                rebase(
                    &mut v.MostDetailedMip,
                    &mut v.MipLevels,
                    &mut v.ResourceMinLODClamp,
                );
            }
            _ => {}
        }
    }
}

/// Synthesizes a full SRV description matching `rd` with concrete `format` when
/// the host texture is typeless and the title passed a null description.
fn default_srv_desc(
    rd: &D3D12_RESOURCE_DESC,
    format: DXGI_FORMAT,
) -> Option<D3D12_SHADER_RESOURCE_VIEW_DESC> {
    let mut desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: format,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        ..Default::default()
    };
    match rd.Dimension {
        D3D12_RESOURCE_DIMENSION_TEXTURE1D => {
            if rd.DepthOrArraySize > 1 {
                desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1DARRAY;
                desc.Anonymous.Texture1DArray = D3D12_TEX1D_ARRAY_SRV {
                    MipLevels: u32::MAX,
                    ArraySize: u32::from(rd.DepthOrArraySize),
                    ..Default::default()
                };
            } else {
                desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1D;
                desc.Anonymous.Texture1D = D3D12_TEX1D_SRV {
                    MipLevels: u32::MAX,
                    ..Default::default()
                };
            }
        }
        D3D12_RESOURCE_DIMENSION_TEXTURE2D => {
            if rd.SampleDesc.Count > 1 {
                if rd.DepthOrArraySize > 1 {
                    desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY;
                    desc.Anonymous.Texture2DMSArray = D3D12_TEX2DMS_ARRAY_SRV {
                        ArraySize: u32::from(rd.DepthOrArraySize),
                        ..Default::default()
                    };
                } else {
                    desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMS;
                    desc.Anonymous.Texture2DMS = D3D12_TEX2DMS_SRV::default();
                }
            } else if rd.DepthOrArraySize > 1 {
                desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
                desc.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_SRV {
                    MipLevels: u32::MAX,
                    ArraySize: u32::from(rd.DepthOrArraySize),
                    ..Default::default()
                };
            } else {
                desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
                desc.Anonymous.Texture2D = D3D12_TEX2D_SRV {
                    MipLevels: u32::MAX,
                    ..Default::default()
                };
            }
        }
        D3D12_RESOURCE_DIMENSION_TEXTURE3D => {
            desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
            desc.Anonymous.Texture3D = D3D12_TEX3D_SRV {
                MipLevels: u32::MAX,
                ..Default::default()
            };
        }
        _ => return None,
    }
    Some(desc)
}

/// Synthesizes a full UAV description matching `rd` with concrete `format` when
/// the host texture is typeless and the title passed a null description.
/// Returns `None` for multisampled textures (invalid for UAVs in D3D12).
fn default_uav_desc(
    rd: &D3D12_RESOURCE_DESC,
    format: DXGI_FORMAT,
) -> Option<D3D12_UNORDERED_ACCESS_VIEW_DESC> {
    let mut desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: format,
        ..Default::default()
    };
    match rd.Dimension {
        D3D12_RESOURCE_DIMENSION_TEXTURE1D => {
            if rd.DepthOrArraySize > 1 {
                desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
                desc.Anonymous.Texture1DArray = D3D12_TEX1D_ARRAY_UAV {
                    ArraySize: u32::from(rd.DepthOrArraySize),
                    ..Default::default()
                };
            } else {
                desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
                desc.Anonymous.Texture1D = D3D12_TEX1D_UAV::default();
            }
        }
        D3D12_RESOURCE_DIMENSION_TEXTURE2D => {
            if rd.SampleDesc.Count > 1 {
                return None;
            }
            if rd.DepthOrArraySize > 1 {
                desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
                desc.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV {
                    ArraySize: u32::from(rd.DepthOrArraySize),
                    ..Default::default()
                };
            } else {
                desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
                desc.Anonymous.Texture2D = D3D12_TEX2D_UAV::default();
            }
        }
        D3D12_RESOURCE_DIMENSION_TEXTURE3D => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
            desc.Anonymous.Texture3D = D3D12_TEX3D_UAV {
                WSize: u32::MAX,
                ..Default::default()
            };
        }
        _ => return None,
    }
    Some(desc)
}

/// Whether the host can make a shader (or unordered access) view of
/// `texture` in `format` (none: the texture's own). The console allows any
/// view; the host refuses (and removes the device for) views without the
/// resource's permission, of typed depth, and in another format than a
/// typed resource's, but for casts it allows (see [`casts_typed`]). A view
/// of a typed texture in another format of its family takes the texture's
/// format, unless it is the texture's sRGB or linear twin and the host
/// casts: then the hardware converts (or not) as the console does.
/// shortcut: other casts within a family (UINT of UNORM, say) read as the
/// texture's own format.
fn fits(texture: &ID3D12Resource, mut format: Option<&mut DXGI_FORMAT>, unordered: bool) -> bool {
    use crate::resource::{DEPTH_FORMATS, family};
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let allowed = if unordered {
        d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0
    } else {
        d.Flags.0 & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 == 0
    };
    let own = d.Format.0 as u32;
    if !allowed || DEPTH_FORMATS.iter().any(|&(typed, _)| typed == own) {
        once!(
            unordered,
            format = own,
            flags = d.Flags.0,
            "view the host does not allow"
        );
        return false;
    }
    // Every family has more than one format, its typeless one first.
    let typeless = is_typeless(own);
    if let Some(format) = format.as_deref_mut()
        && unordered
        && let Some(&(_, linear)) = crate::resource::SRGB
            .iter()
            .find(|&&(s, _)| s == format.0 as u32)
    {
        format.0 = linear as i32;
    }
    let view = format.as_ref().map_or(0, |f| f.0 as u32);
    // Typeless host copies need an explicit concrete format: `make` fills
    // null / UNKNOWN via `concrete_format_for` before calling here (VIC-BT-004).
    if view == 0 {
        return !typeless;
    }
    // A texture whose console format PC forbids as a render target or UAV
    // was substituted (`resource::target_format`): views name its host format.
    if view != own && crate::resource::target_format(view) == own {
        if let Some(format) = format {
            *format = d.Format;
        }
        return true;
    }
    if family(view) != family(own) {
        once!(view, own, "view format of another family");
        return false;
    }
    let twin = crate::resource::SRGB
        .iter()
        .any(|&(s, l)| (s, l) == (view, own) || (l, s) == (view, own));
    if let Some(format) = format
        && !typeless
        && !(twin && casts_typed())
    {
        *format = d.Format;
    }
    true
}

/// The unordered access view description for a title's view of a depth
/// texture, made for the texture's colour twin `twin`
/// ([`crate::resource::Twin`]): the same mip and slices, in the colour
/// format of the twin's family for a typeless or depth view format.
/// The twin of a stencil plane (R8) stands for the views of that plane
/// only: those that name plane 1 or a stencil format. A view of the depth
/// plane has a twin of its own, viewed as a depth texture's without
/// stencil: a view of it that names plane 0 or a depth format takes the
/// twin's colour format too.
fn twin_uav(
    twin: &ID3D12Resource,
    desc: Option<&D3D12_UNORDERED_ACCESS_VIEW_DESC>,
) -> Option<D3D12_UNORDERED_ACCESS_VIEW_DESC> {
    // SAFETY: a live resource.
    let d = unsafe { twin.GetDesc() };
    let mut desc = desc.cloned().unwrap_or_else(|| {
        let mut all = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        if d.DepthOrArraySize > 1 {
            all.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            all.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV {
                ArraySize: u32::from(d.DepthOrArraySize),
                ..Default::default()
            };
        }
        all
    });
    // A stencil plane's twin (R8 typeless, 60): the title's view is of the
    // stencil plane if it names plane 1 or a stencil view format (22, 47).
    if d.Format.0 == 60 {
        let mut plane = 0;
        match desc.ViewDimension {
            D3D12_UAV_DIMENSION_TEXTURE2D => {
                // SAFETY: the member the dimension names.
                let mut view = unsafe { desc.Anonymous.Texture2D };
                plane = std::mem::take(&mut view.PlaneSlice);
                desc.Anonymous.Texture2D = view;
            }
            D3D12_UAV_DIMENSION_TEXTURE2DARRAY => {
                // SAFETY: the member the dimension names.
                let mut view = unsafe { desc.Anonymous.Texture2DArray };
                plane = std::mem::take(&mut view.PlaneSlice);
                desc.Anonymous.Texture2DArray = view;
            }
            _ => {}
        }
        if plane != 1 && !matches!(desc.Format.0, 22 | 47) {
            return None;
        }
        desc.Format.0 = 62;
        return Some(desc);
    }
    // None, typeless and depth formats (R16: 0, 53, 55; R32: 0, 39, 40),
    // and those of a depth-stencil format's depth plane (19 to 21, 44 to
    // 46), for the twin of such a plane: the twin's UNORM / float view
    // formats (R16_UNORM 56, R32_FLOAT 41).
    desc.Format.0 = match (desc.Format.0, d.Format.0) {
        (0 | 53 | 55 | 19..=21 | 44..=46, 53) => 56,
        (0 | 39 | 40 | 19..=21 | 44..=46, 39) => 41,
        (other, _) => other,
    };
    Some(desc)
}

/// Makes the host descriptor for `view` at `at`: the host resource it
/// views, or `None` if it cannot.
fn make(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> Option<Kept> {
    let host = host()?;
    let device = &host.device;
    match view {
        // make_at and the mirror path write a typed null for this view.
        View::Constant(0, _) => None,
        &View::Constant(address, size) => {
            let shadow = crate::shadow::find(address, u64::from(size))?;
            let gpu = shadow.gpu(address);
            if gpu % 256 != 0 {
                once!(
                    address = format_args!("{address:#x}"),
                    "constant buffer not 256-byte aligned on the host"
                );
                return None;
            }
            let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
                BufferLocation: gpu,
                SizeInBytes: size.next_multiple_of(256),
            };
            // SAFETY: a slot of our heap; the range lies in the host copy.
            unsafe { device.CreateConstantBufferView(Some(&desc), at) };
            Some(Kept::Shadow(shadow))
        }
        View::Shader(resource, desc) => {
            let resource = *resource as crate::object::This;
            if let Some(texture) = crate::resource::host_resource(resource) {
                // SAFETY: a live resource.
                let rd = unsafe { texture.GetDesc() };
                let own = rd.Format.0 as u32;
                let mut desc = desc.as_ref().map(srv_desc);
                if is_typeless(own) {
                    match desc.as_mut() {
                        Some(dsc) if dsc.Format == DXGI_FORMAT_UNKNOWN => {
                            dsc.Format = concrete_format_for(resource, own, false)?;
                        }
                        Some(_) => {}
                        None => {
                            let fmt = concrete_format_for(resource, own, false)?;
                            desc = Some(default_srv_desc(&rd, fmt)?);
                        }
                    }
                }
                if !fits(texture, desc.as_mut().map(|d| &mut d.Format), false) {
                    return None;
                }
                if let Some(dsc) = desc.as_mut() {
                    drop_mips_in_srv(dsc, crate::resource::dropped_mips(resource));
                }
                // A texture only part of which is mapped (streaming in):
                // sample the mips that are there, not the zeros of the ones
                // that are not. The console streams a texture's mips into
                // place and clamps the level of detail to what arrived; the
                // host has no clamp but the view's window. Remember the
                // narrowed window so a later mirror can widen it.
                let host_mips = desc
                    .as_mut()
                    .and_then(|d| narrow_to_resident(resource, texture, d));
                // SAFETY: a live texture and a slot of our heap.
                unsafe {
                    device.CreateShaderResourceView(
                        texture,
                        desc.as_ref().map(std::ptr::from_ref),
                        at,
                    )
                };
                Some(Kept::Texture {
                    resource: texture.clone(),
                    title: resource as usize,
                    is_uav: false,
                    host_mips,
                })
            } else if let (Some((data, len)), Some(desc)) =
                (crate::resource::buffer_bytes(resource), desc)
            {
                buffer_srv(data as u64, len, srv_desc(desc), at)
            } else {
                None
            }
        }
        View::Unordered(resource, _counter, desc) => {
            let resource = *resource as crate::object::This;
            let Some(texture) = crate::resource::host_resource(resource) else {
                return match (crate::resource::buffer_bytes(resource), desc) {
                    (Some((data, len)), Some(desc)) => {
                        buffer_uav(data as u64, len, uav_desc(desc), at)
                    }
                    _ => None,
                };
            };
            // SAFETY: a live resource.
            let rd = unsafe { texture.GetDesc() };
            let own = rd.Format.0 as u32;
            let mut desc = desc.as_ref().map(uav_desc);
            if is_typeless(own) {
                match desc.as_mut() {
                    Some(dsc) if dsc.Format == DXGI_FORMAT_UNKNOWN => {
                        dsc.Format = concrete_format_for(resource, own, true)?;
                    }
                    Some(_) => {}
                    None => {
                        let fmt = concrete_format_for(resource, own, true)?;
                        desc = Some(default_uav_desc(&rd, fmt)?);
                    }
                }
            }
            // A depth texture's unordered access is its colour twin's: the
            // host gives a depth texture none.
            let twin = crate::resource::twin_resource(resource)
                .and_then(|twin| Some((twin, twin_uav(twin, desc.as_ref())?)))
                // A view of the depth plane of a texture with a stencil
                // plane: of that plane's own twin.
                .or_else(|| {
                    let twin = crate::resource::twin_plane_resource(resource)?;
                    Some((twin, twin_uav(twin, desc.as_ref())?))
                });
            let texture = match twin {
                Some((twin, view)) => {
                    desc = Some(view);
                    twin
                }
                None => texture,
            };
            if !fits(texture, desc.as_mut().map(|d| &mut d.Format), true) {
                return None;
            }
            // SAFETY: a live texture and a slot of our heap.
            unsafe {
                device.CreateUnorderedAccessView(
                    texture,
                    None::<&ID3D12Resource>,
                    desc.as_ref().map(std::ptr::from_ref),
                    at,
                )
            };
            Some(Kept::Texture {
                resource: texture.clone(),
                title: resource as usize,
                is_uav: true,
                host_mips: None,
            })
        }
        View::Sampler(words) => {
            // SAFETY: 13 words in the host's layout.
            let desc = unsafe {
                std::ptr::from_ref(words)
                    .cast::<D3D12_SAMPLER_DESC>()
                    .read_unaligned()
            };
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(Kept::Empty)
        }
        View::ConsoleSampler(words) => {
            let desc = crate::sampler::console(words);
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(Kept::Empty)
        }
        View::Placed(unordered, resource, desc) => {
            if let Some(view) = placed_texture(view) {
                return make(&view, at);
            }
            const BUFFER: u32 = 1;
            if resource[0] != BUFFER {
                // A texture the title placed itself with no host resource at
                // its address stays unmirrored: null, as before.
                once!(
                    dimension = resource[0],
                    unordered = *unordered,
                    "placed texture view not mirrored"
                );
                return None;
            }
            // The resource description's width, after the alignment.
            let len = u64::from(resource[4]) | u64::from(resource[5]) << 32;
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            let words: &[u32; 10] = desc[..10].try_into().unwrap_or(&[0; 10]);
            if *unordered {
                buffer_uav(address, len, uav_desc(words), at)
            } else {
                buffer_srv(address, len, srv_desc(words), at)
            }
        }
    }
}

/// First mip sampled by a shader view the title placed over a single 2D texture.
pub(crate) fn placed_first_mip(view: &View) -> u32 {
    let View::Placed(false, _, words) = view else {
        return 0;
    };
    let desc = srv_desc(words[..10].try_into().unwrap());
    if desc.ViewDimension == D3D12_SRV_DIMENSION_TEXTURE2D {
        // SAFETY: the dimension selects the Texture2D union member.
        unsafe { desc.Anonymous.Texture2D.MostDetailedMip }
    } else {
        0
    }
}

/// Resolves a placed texture's SRV or UAV to the resource created at its
/// address (the resource is made on first use).
pub(crate) fn placed_texture(view: &View) -> Option<View> {
    const BUFFER: u32 = 1;
    let View::Placed(unordered, resource, desc) = view else {
        return None;
    };
    if resource[0] == BUFFER {
        return None;
    }
    let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
    let texture = crate::resource::texture_at(address, resource)? as usize;
    let desc = desc[..10].try_into().ok();
    Some(if *unordered {
        View::Unordered(texture, 0, desc)
    } else {
        View::Shader(texture, desc)
    })
}

/// A null view of the kind `view` is, for a slot whose view the host
/// cannot make: unordered access (to a buffer or a 2D texture) or a shader
/// view. Writes to it are dropped, reads give zeros.
fn null_for(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let is_buffer = |r: &usize| crate::resource::buffer_bytes(*r as crate::object::This).is_some();
    // Unordered views: whether a buffer, and the title's view dimension.
    let (buffer, dimension) = match view {
        View::Unordered(r, _, desc) => (
            is_buffer(r) || desc.is_some_and(|d| d[1] == 1),
            desc.map(|d| d[1]),
        ),
        View::Placed(true, resource, desc) => (resource[0] == 1, Some(desc[1])),
        // Shader views of buffers read nothing through a null buffer view:
        // a null 2D view where the shader expects a buffer is a mismatch
        // the hardware does not survive in general (GPU-based validation
        // reports it).
        View::Shader(r, desc) if is_buffer(r) || desc.is_some_and(|d| d[1] == 1) => {
            return null_buffer_view(at);
        }
        View::Placed(false, resource, _) if resource[0] == 1 => return null_buffer_view(at),
        // Textures: a null view of the title's dimension (1D, arrays, 3D,
        // cubes), not always 2D.
        // shortcut: the title's null SRV at heap slot 0 is one of these
        // (resource 0 with a texture desc). Slot 0 is also read as a CBV
        // through a computed heap index; a single host descriptor cannot be
        // both an SRV and a CBV, so the translator skips heap-direct CBV
        // loads at index 0 rather than asking for a null CBV here (#485).
        View::Shader(_, Some(desc)) => return null_like(desc, at),
        View::Placed(false, _, desc) => {
            return null_like(desc[..10].try_into().unwrap_or(&[0; 10]), at);
        }
        View::Constant(..) => {
            let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC::default();
            // SAFETY: a slot of our heap; a zero location makes a null view.
            return unsafe { host.device.CreateConstantBufferView(Some(&desc), at) };
        }
        _ => return null_view(at),
    };
    let desc = if buffer {
        D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            ..Default::default()
        }
    } else {
        null_uav_desc(dimension)
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device.CreateUnorderedAccessView(
            None::<&ID3D12Resource>,
            None::<&ID3D12Resource>,
            Some(&desc),
            at,
        )
    };
}

/// A null texture UAV of the title's view dimension (1D, 1D or 2D arrays,
/// 3D), 2D if it names none: a 2D null view where the shader writes a 3D
/// texture is a mismatch GPU-based validation reports (the compute shader
/// that copies the world's wind field, 2026-11-24).
fn null_uav_desc(dimension: Option<u32>) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_UINT,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        ..Default::default()
    };
    match dimension.map(|d| D3D12_UAV_DIMENSION(d as i32)) {
        Some(D3D12_UAV_DIMENSION_TEXTURE1D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE1DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
            desc.Anonymous.Texture1DArray = D3D12_TEX1D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE2DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            desc.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE3D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
            desc.Anonymous.Texture3D = D3D12_TEX3D_UAV {
                WSize: u32::MAX,
                ..Default::default()
            };
        }
        _ => {}
    }
    desc
}

/// Raw views can expose only whole four-byte elements. A trailing partial
/// element belongs outside the view, even when it is the entire range.
fn raw_element_count(len: u64) -> Option<std::num::NonZeroU32> {
    std::num::NonZeroU32::new(u32::try_from(len / 4).unwrap_or(u32::MAX))
}

/// A buffer view that reads zeros (raw, as translated shaders mostly read
/// buffers).
fn null_buffer_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_SRV {
                NumElements: 1,
                Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A texture view that reads zeros with the dimension (and mip and array
/// counts) of the title's description `words`; 2D if it names none.
fn null_like(words: &[u32; 10], at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    const TEXTURES: std::ops::RangeInclusive<u32> = 2..=10;
    if !TEXTURES.contains(&words[1]) {
        return null_view(at);
    }
    let Some(host) = host() else { return };
    // One mip, one slice or cube, plane 0: fields a null view still checks.
    let mut w = [0u32; 10];
    w[1] = words[1];
    match words[1] {
        6 => {}
        7 => w[5] = 1,
        3 | 5 | 10 => (w[5], w[7]) = (1, 1),
        _ => w[5] = 1,
    }
    let mut desc = srv_desc(&w);
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A view that reads zeros, for slots whose view the host cannot make.
/// shortcut: slots with no view at all (kind unknown) get this null 2D
/// view; null buffer views there did not stop the world's low-address
/// page faults (2026-11-24, run w10), so they are not the cause.
fn null_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MipLevels: 1,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_raw_view_counts_only_complete_dwords() {
        let count = |bytes| raw_element_count(bytes).map(std::num::NonZeroU32::get);
        for bytes in 0..4 {
            assert_eq!(count(bytes), None);
        }
        for bytes in 4..8 {
            assert_eq!(count(bytes), Some(1));
        }
        assert_eq!(count(8), Some(2));
        assert_eq!(count(u64::MAX), Some(u32::MAX));
    }

    /// The CPU-side block used to initialize null samplers must fit the
    /// same portable sampler limit as the shader-visible heap.
    #[test]
    fn portable_sampler_null_fill_fits_the_host_limit() {
        if host().is_none() {
            return; // no host GPU
        }
        let heap = HostHeap::new(true, D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE)
            .expect("portable sampler heap must initialize");
        assert!(
            heap.fill_null(
                D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
                D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE,
            )
            .is_ok()
        );
    }

    /// A null texture UAV keeps the title's dimension (3D for the world's
    /// wind field), and is 2D when the title names none.
    #[test]
    fn null_uavs_keep_the_titles_dimension() {
        let d = null_uav_desc(Some(8));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE3D);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture3D.WSize }, u32::MAX);
        let d = null_uav_desc(Some(5));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE2DARRAY);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture2DArray.ArraySize }, 1);
        let dimension = |d| null_uav_desc(d).ViewDimension;
        assert_eq!(dimension(Some(2)), D3D12_UAV_DIMENSION_TEXTURE1D);
        assert_eq!(dimension(Some(4)), D3D12_UAV_DIMENSION_TEXTURE2D);
        assert_eq!(dimension(None), D3D12_UAV_DIMENSION_TEXTURE2D);
        // Multisampled UAVs need a cap; they stay 2D.
        assert_eq!(dimension(Some(6)), D3D12_UAV_DIMENSION_TEXTURE2D);
    }

    /// A null 3D UAV made on the real device leaves it alive.
    #[test]
    fn null_3d_uav_is_accepted_by_the_device() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let view = View::Unordered(0, 0, Some([28, 8, 0, 0, 0, 0, 0, 0, 0, 0]));
        null_for(&view, heap.cpu(0));
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// The title's null SRV at heap slot 0 is a texture view of no resource.
    /// It must stay an SRV (not a null CBV): slot 0 is also sampled as an
    /// SRV, and a single host descriptor cannot satisfy both kinds (#485).
    #[test]
    fn title_null_srv_at_slot_zero_stays_an_srv() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        // Format R32_UINT, TEXTURE1D, default component mapping — the shape
        // reported for the title's null SRV at heap slot 0.
        let view = View::Shader(
            0,
            Some([
                42,
                2,
                D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
            ]),
        );
        null_for(&view, heap.cpu(0));
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// Buffer views of no elements (the title makes them for buffers it has
    /// not sized yet) are left null: the host removes the device for one.
    #[test]
    fn buffer_views_of_no_elements_are_left_null() {
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_UINT;
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let memory = vec![0u32; 1024];
        let address = memory.as_ptr() as u64;
        let resource = 0x7e57_0000usize;
        crate::shadow::register(resource, address, 4096);
        let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV::default(),
            },
        };
        assert!(buffer_uav(address, 4096, uav, heap.cpu(0)).is_none());
        let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV::default(),
            },
        };
        assert!(buffer_srv(address, 4096, srv, heap.cpu(1)).is_none());
        crate::shadow::unregister(resource, address);
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// Dynamic slots stay taken until given back, and runs do not wrap.
    #[test]
    fn dynamic_slots_are_held_until_released() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        let a = heap.dynamic(heap.spare - 100).unwrap();
        let b = heap.dynamic(60).unwrap();
        assert_eq!((a, b), (4, 4 + heap.spare - 100));
        // 40 left at the end: 50 do not fit until something is given back.
        assert_eq!(heap.dynamic(50), None);
        heap.release(a, 50);
        assert_eq!(heap.dynamic(50), Some(4));
        assert_eq!(heap.dynamic(40), Some(4 + heap.spare - 40));
    }

    /// A sampler heap past the host's limit is capped, not lost, and keeps
    /// dynamic slots for the tables reaching past what it holds.
    #[test]
    fn large_sampler_heap_is_capped() {
        let Some(heap) = HostHeap::new(true, 4 * SAMPLERS) else {
            return; // no host GPU
        };
        let count = SAMPLERS - SAMPLER_DYNAMIC;
        assert_eq!(heap.count(), count);
        assert_eq!(heap.dynamic(SAMPLER_DYNAMIC + 1), None);
        let a = heap.dynamic(SAMPLER_DYNAMIC).unwrap();
        assert_eq!(a, count);
        assert_eq!(heap.dynamic(1), None);
        heap.release(a, 8);
        assert_eq!(heap.dynamic(8), Some(count));
    }

    /// A sampler heap within the limit has a slot per title slot and no
    /// dynamic ones.
    #[test]
    fn small_sampler_heap_has_no_dynamic_slots() {
        let Some(heap) = HostHeap::new(true, 16) else {
            return; // no host GPU
        };
        assert_eq!(heap.count(), 16);
        assert_eq!(heap.dynamic(1), None);
    }

    /// A structured buffer SRV whose offset inside its shadow buffer is not a
    /// multiple of its element stride reads its own aligned host copy, refreshed
    /// from title memory when a command list referencing the slot is submitted.
    #[test]
    fn unaligned_structured_buffer_srv_refreshes_on_submission() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let mut backing: Vec<u8> = (0..4096usize).map(|i| (i % 251) as u8).collect();
        let base = backing.as_ptr() as u64;
        crate::shadow::register(base as usize, base, backing.len() as u64);
        // Offset 32 is not a multiple of stride 24 (32 % 24 == 8).
        let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_UNKNOWN,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: 0,
                    NumElements: 1,
                    StructureByteStride: 24,
                    Flags: D3D12_BUFFER_SRV_FLAG_NONE,
                },
            },
        };
        let kept = buffer_srv(base + 32, 24, desc, heap.cpu(0))
            .expect("unaligned structured SRV should succeed");
        let (copy, start, len) = match &kept {
            Kept::AliasedShadow {
                shadow,
                address,
                len,
            } => (shadow.clone(), *address, *len),
            _ => panic!("expected an aligned view copy"),
        };
        heap.keep(0, kept);

        // Mutate the title slice *after* creating the view (as a ring buffer does).
        let updated: [u8; 24] = std::array::from_fn(|i| 0xA0u8.wrapping_add(i as u8));
        backing[32..56].copy_from_slice(&updated);

        let list = host.new_list().expect("host command list");
        heap.refresh_slot(0, &list);
        list.close();
        host.execute(&[&list]);
        crate::shadow::unregister(base as usize, base);

        let got = copy.bytes(start, len).expect("view bytes inside its copy");
        assert_eq!(got, &updated);
        // SAFETY: queries the live host device's removal status only.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    struct OwnedObject(crate::object::This);

    impl Drop for OwnedObject {
        fn drop(&mut self) {
            if !self.0.is_null() {
                let release = crate::object::slot::<extern "system" fn(crate::object::This) -> u32>(
                    self.0, 0x10,
                );
                release(self.0);
            }
        }
    }

    fn make_texture(
        dimension: D3D12_RESOURCE_DIMENSION,
        width: u64,
        height: u32,
        depth_or_array: u16,
        format: DXGI_FORMAT,
        sample_count: u32,
        flags: D3D12_RESOURCE_FLAGS,
    ) -> OwnedObject {
        use windows::core::Interface;
        let null = std::ptr::null();
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: dimension,
            Width: width,
            Height: height,
            DepthOrArraySize: depth_or_array,
            MipLevels: 1,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: sample_count,
                Quality: 0,
            },
            Flags: flags,
            ..Default::default()
        };
        let iid = ID3D12Resource::IID;
        let mut out = null;
        assert_eq!(
            crate::resource::create_committed(
                null,
                (&raw const heap).cast(),
                0,
                (&raw const desc).cast(),
                0,
                null.cast(),
                (&raw const iid).cast(),
                &mut out,
            ),
            crate::object::S_OK
        );
        OwnedObject(out)
    }

    fn make_view_heap(count: u32) -> OwnedObject {
        use windows::core::Interface;
        #[repr(C)]
        struct ViewHeap {
            desc: D3D12_DESCRIPTOR_HEAP_DESC,
            placement: [u64; 2],
        }
        let desc = ViewHeap {
            desc: D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: count,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            },
            placement: [0; 2],
        };
        let iid = ID3D12DescriptorHeap::IID;
        let mut out = std::ptr::null();
        assert_eq!(
            crate::descriptor::create_heap(
                std::ptr::null(),
                (&raw const desc).cast(),
                (&raw const iid).cast(),
                &mut out,
            ),
            crate::object::S_OK
        );
        OwnedObject(out)
    }

    #[test]
    fn default_concrete_formats_cover_typeless_families() {
        let srv = |f: DXGI_FORMAT| default_concrete_format(f.0 as u32, false);
        let uav = |f: DXGI_FORMAT| default_concrete_format(f.0 as u32, true);
        // Multi-plane depth-stencil: SRV only.
        assert_eq!(
            srv(DXGI_FORMAT_R24G8_TYPELESS),
            Some(DXGI_FORMAT_R24_UNORM_X8_TYPELESS.0 as u32)
        );
        assert_eq!(uav(DXGI_FORMAT_R24G8_TYPELESS), None);
        assert_eq!(
            srv(DXGI_FORMAT_R32G8X24_TYPELESS),
            Some(DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS.0 as u32)
        );
        assert_eq!(uav(DXGI_FORMAT_R32G8X24_TYPELESS), None);
        // Block-compressed: SRV only.
        for (typeless, concrete) in [
            (DXGI_FORMAT_BC1_TYPELESS, DXGI_FORMAT_BC1_UNORM),
            (DXGI_FORMAT_BC2_TYPELESS, DXGI_FORMAT_BC2_UNORM),
            (DXGI_FORMAT_BC3_TYPELESS, DXGI_FORMAT_BC3_UNORM),
            (DXGI_FORMAT_BC4_TYPELESS, DXGI_FORMAT_BC4_UNORM),
            (DXGI_FORMAT_BC5_TYPELESS, DXGI_FORMAT_BC5_UNORM),
            (DXGI_FORMAT_BC6H_TYPELESS, DXGI_FORMAT_BC6H_UF16),
            (DXGI_FORMAT_BC7_TYPELESS, DXGI_FORMAT_BC7_UNORM),
        ] {
            assert_eq!(srv(typeless), Some(concrete.0 as u32));
            assert_eq!(uav(typeless), None);
        }
        // 96-bit RGB: SRV only (D3D12 has no typed UAV for R32G32B32_FLOAT).
        assert_eq!(
            srv(DXGI_FORMAT_R32G32B32_TYPELESS),
            Some(DXGI_FORMAT_R32G32B32_FLOAT.0 as u32)
        );
        assert_eq!(uav(DXGI_FORMAT_R32G32B32_TYPELESS), None);
        // Color typeless: always valid as SRV, gated by host typed-UAV support as UAV.
        for (typeless, concrete) in [
            (
                DXGI_FORMAT_R32G32B32A32_TYPELESS,
                DXGI_FORMAT_R32G32B32A32_FLOAT,
            ),
            (
                DXGI_FORMAT_R16G16B16A16_TYPELESS,
                DXGI_FORMAT_R16G16B16A16_FLOAT,
            ),
            (DXGI_FORMAT_R32G32_TYPELESS, DXGI_FORMAT_R32G32_FLOAT),
            (
                DXGI_FORMAT_R10G10B10A2_TYPELESS,
                DXGI_FORMAT_R10G10B10A2_UNORM,
            ),
            (DXGI_FORMAT_R8G8B8A8_TYPELESS, DXGI_FORMAT_R8G8B8A8_UNORM),
            (DXGI_FORMAT_R16G16_TYPELESS, DXGI_FORMAT_R16G16_FLOAT),
            (DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_R32_FLOAT),
            (DXGI_FORMAT_R8G8_TYPELESS, DXGI_FORMAT_R8G8_UNORM),
            (DXGI_FORMAT_R16_TYPELESS, DXGI_FORMAT_R16_FLOAT),
            (DXGI_FORMAT_R8_TYPELESS, DXGI_FORMAT_R8_UNORM),
            (DXGI_FORMAT_B8G8R8A8_TYPELESS, DXGI_FORMAT_B8G8R8A8_UNORM),
            (DXGI_FORMAT_B8G8R8X8_TYPELESS, DXGI_FORMAT_B8G8R8X8_UNORM),
        ] {
            assert_eq!(srv(typeless), Some(concrete.0 as u32));
            assert_eq!(
                uav(typeless),
                supports_typed_uav(concrete).then_some(concrete.0 as u32)
            );
        }
        assert_eq!(srv(DXGI_FORMAT_R8G8B8A8_UNORM), None);
    }

    #[test]
    fn default_view_descs_match_resource_dimensions() {
        let base = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 8,
            Height: 8,
            DepthOrArraySize: 1,
            MipLevels: 3,
            Format: DXGI_FORMAT_R8G8B8A8_TYPELESS,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            ..Default::default()
        };
        let fmt = DXGI_FORMAT_R8G8B8A8_UNORM;

        let d1 = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE1D,
            Height: 1,
            ..base
        };
        assert_eq!(
            default_srv_desc(&d1, fmt).unwrap().ViewDimension,
            D3D12_SRV_DIMENSION_TEXTURE1D
        );
        assert_eq!(
            default_uav_desc(&d1, fmt).unwrap().ViewDimension,
            D3D12_UAV_DIMENSION_TEXTURE1D
        );

        let d1a = D3D12_RESOURCE_DESC {
            DepthOrArraySize: 3,
            ..d1
        };
        let srv = default_srv_desc(&d1a, fmt).unwrap();
        assert_eq!(srv.ViewDimension, D3D12_SRV_DIMENSION_TEXTURE1DARRAY);
        // SAFETY: Texture1DArray matches ViewDimension.
        assert_eq!(unsafe { srv.Anonymous.Texture1DArray.ArraySize }, 3);
        let uav = default_uav_desc(&d1a, fmt).unwrap();
        assert_eq!(uav.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE1DARRAY);
        // SAFETY: Texture1DArray matches ViewDimension.
        assert_eq!(unsafe { uav.Anonymous.Texture1DArray.ArraySize }, 3);

        assert_eq!(
            default_srv_desc(&base, fmt).unwrap().ViewDimension,
            D3D12_SRV_DIMENSION_TEXTURE2D
        );
        assert_eq!(
            default_uav_desc(&base, fmt).unwrap().ViewDimension,
            D3D12_UAV_DIMENSION_TEXTURE2D
        );

        let d2a = D3D12_RESOURCE_DESC {
            DepthOrArraySize: 4,
            ..base
        };
        let srv = default_srv_desc(&d2a, fmt).unwrap();
        assert_eq!(srv.ViewDimension, D3D12_SRV_DIMENSION_TEXTURE2DARRAY);
        // SAFETY: Texture2DArray matches ViewDimension.
        assert_eq!(unsafe { srv.Anonymous.Texture2DArray.ArraySize }, 4);
        let uav = default_uav_desc(&d2a, fmt).unwrap();
        assert_eq!(uav.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE2DARRAY);
        // SAFETY: Texture2DArray matches ViewDimension.
        assert_eq!(unsafe { uav.Anonymous.Texture2DArray.ArraySize }, 4);

        let d2ms = D3D12_RESOURCE_DESC {
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 4,
                Quality: 0,
            },
            ..base
        };
        assert_eq!(
            default_srv_desc(&d2ms, fmt).unwrap().ViewDimension,
            D3D12_SRV_DIMENSION_TEXTURE2DMS
        );
        assert!(default_uav_desc(&d2ms, fmt).is_none());

        let d2msa = D3D12_RESOURCE_DESC {
            DepthOrArraySize: 2,
            ..d2ms
        };
        let srv = default_srv_desc(&d2msa, fmt).unwrap();
        assert_eq!(srv.ViewDimension, D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY);
        // SAFETY: Texture2DMSArray matches ViewDimension.
        assert_eq!(unsafe { srv.Anonymous.Texture2DMSArray.ArraySize }, 2);
        assert!(default_uav_desc(&d2msa, fmt).is_none());

        let d3 = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE3D,
            DepthOrArraySize: 5,
            ..base
        };
        let srv = default_srv_desc(&d3, fmt).unwrap();
        assert_eq!(srv.ViewDimension, D3D12_SRV_DIMENSION_TEXTURE3D);
        // SAFETY: Texture3D matches ViewDimension.
        assert_eq!(unsafe { srv.Anonymous.Texture3D.MipLevels }, u32::MAX);
        let uav = default_uav_desc(&d3, fmt).unwrap();
        assert_eq!(uav.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE3D);
        // SAFETY: Texture3D matches ViewDimension.
        assert_eq!(unsafe { uav.Anonymous.Texture3D.WSize }, u32::MAX);
    }

    #[test]
    fn typeless_default_srvs_and_uavs_create_cleanly_on_device() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let _ = host.log_messages();
        let Some(heap) = HostHeap::new(false, 32) else {
            return;
        };
        let mut slot = 0u32;
        let mut next_slot = || {
            let s = slot;
            slot += 1;
            heap.cpu(s)
        };

        let rgba_typeless = make_texture(
            D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            4,
            4,
            1,
            DXGI_FORMAT_R8G8B8A8_TYPELESS,
            1,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        );
        let r32_typeless = make_texture(
            D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            4,
            4,
            1,
            DXGI_FORMAT_R32_TYPELESS,
            1,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        );
        let bgra_typeless = make_texture(
            D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            4,
            4,
            1,
            DXGI_FORMAT_B8G8R8A8_TYPELESS,
            1,
            D3D12_RESOURCE_FLAG_NONE,
        );
        let r24g8_typeless = make_texture(
            D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            4,
            4,
            1,
            DXGI_FORMAT_R24G8_TYPELESS,
            1,
            D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
        );
        let bc7_typeless = make_texture(
            D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            4,
            4,
            1,
            DXGI_FORMAT_BC7_TYPELESS,
            1,
            D3D12_RESOURCE_FLAG_NONE,
        );
        let msaa_typeless = make_texture(
            D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            4,
            4,
            1,
            DXGI_FORMAT_R8G8B8A8_TYPELESS,
            4,
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
        );
        let srgb_uav = make_texture(
            D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            4,
            4,
            1,
            DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
            1,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        );
        let d24s8 = make_texture(
            D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            4,
            4,
            1,
            DXGI_FORMAT_D24_UNORM_S8_UINT,
            1,
            D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
        );
        let d16 = make_texture(
            D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            4,
            4,
            1,
            DXGI_FORMAT_D16_UNORM,
            1,
            D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
        );
        let tex3d = make_texture(
            D3D12_RESOURCE_DIMENSION_TEXTURE3D,
            4,
            4,
            2,
            DXGI_FORMAT_R8G8B8A8_TYPELESS,
            1,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        );
        let tex1d_arr = make_texture(
            D3D12_RESOURCE_DIMENSION_TEXTURE1D,
            4,
            1,
            2,
            DXGI_FORMAT_R8G8B8A8_TYPELESS,
            1,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        );

        let srv_unknown_2d = [
            DXGI_FORMAT_UNKNOWN.0 as u32,
            D3D12_SRV_DIMENSION_TEXTURE2D.0 as u32,
            D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
        ];
        let uav_unknown_2d = [
            DXGI_FORMAT_UNKNOWN.0 as u32,
            D3D12_UAV_DIMENSION_TEXTURE2D.0 as u32,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ];

        // SRVs (both desc = None and Format = UNKNOWN) succeed.
        for obj in [
            &rgba_typeless,
            &r32_typeless,
            &bgra_typeless,
            &r24g8_typeless,
            &bc7_typeless,
            &srgb_uav,
            &d24s8,
            &d16,
        ] {
            let v_none = View::Shader(obj.0 as usize, None);
            assert!(
                matches!(
                    make(&v_none, next_slot()),
                    Some(Kept::Texture { is_uav: false, .. })
                ),
                "default SRV failed"
            );
            let v_unknown = View::Shader(obj.0 as usize, Some(srv_unknown_2d));
            assert!(
                matches!(
                    make(&v_unknown, next_slot()),
                    Some(Kept::Texture { is_uav: false, .. })
                ),
                "UNKNOWN SRV failed"
            );
        }
        assert!(matches!(
            make(&View::Shader(msaa_typeless.0 as usize, None), next_slot()),
            Some(Kept::Texture { is_uav: false, .. })
        ));
        assert!(matches!(
            make(&View::Shader(tex3d.0 as usize, None), next_slot()),
            Some(Kept::Texture { is_uav: false, .. })
        ));
        assert!(matches!(
            make(&View::Shader(tex1d_arr.0 as usize, None), next_slot()),
            Some(Kept::Texture { is_uav: false, .. })
        ));

        // sRGB+UAV keeps sRGB in default SRVs.
        assert_eq!(
            concrete_format_for(srgb_uav.0, DXGI_FORMAT_R8G8B8A8_TYPELESS.0 as u32, false),
            Some(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB)
        );
        assert_eq!(
            concrete_format_for(srgb_uav.0, DXGI_FORMAT_R8G8B8A8_TYPELESS.0 as u32, true),
            Some(DXGI_FORMAT_R8G8B8A8_UNORM)
        );

        // Valid color UAVs succeed (desc = None and Format = UNKNOWN).
        for obj in [&rgba_typeless, &r32_typeless, &srgb_uav] {
            let v_none = View::Unordered(obj.0 as usize, 0, None);
            assert!(
                matches!(
                    make(&v_none, next_slot()),
                    Some(Kept::Texture { is_uav: true, .. })
                ),
                "default UAV failed"
            );
            let v_unknown = View::Unordered(obj.0 as usize, 0, Some(uav_unknown_2d));
            assert!(
                matches!(
                    make(&v_unknown, next_slot()),
                    Some(Kept::Texture { is_uav: true, .. })
                ),
                "UNKNOWN UAV failed"
            );
        }
        assert!(matches!(
            make(&View::Unordered(tex3d.0 as usize, 0, None), next_slot()),
            Some(Kept::Texture { is_uav: true, .. })
        ));
        assert!(matches!(
            make(&View::Unordered(tex1d_arr.0 as usize, 0, None), next_slot()),
            Some(Kept::Texture { is_uav: true, .. })
        ));

        // Depth, BC, and MSAA UAVs return None and leave null descriptors cleanly.
        for obj in [&r24g8_typeless, &bc7_typeless, &msaa_typeless] {
            for desc in [None, Some(uav_unknown_2d)] {
                let view = View::Unordered(obj.0 as usize, 0, desc);
                let cpu = next_slot();
                assert!(make(&view, cpu).is_none());
                null_for(&view, cpu);
            }
        }

        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
        assert_eq!(host.log_messages(), None, "unexpected D3D12 debug error");
    }

    #[test]
    fn bindless_tracker_handles_over_4096_srvs_overwrites_and_forgets() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let _ = host.log_messages();
        const COUNT: u32 = 5000;
        let heap_obj = make_view_heap(COUNT + 16);
        let heap_ref = crate::descriptor::heap(heap_obj.0);
        let start = heap_ref.start;
        let slot_handle = |i: u32| start + u64::from(i) * 32;

        let mut textures = Vec::with_capacity(COUNT as usize);
        for i in 0..COUNT {
            let tex = make_texture(
                D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                1,
                1,
                1,
                DXGI_FORMAT_R8G8B8A8_UNORM,
                1,
                D3D12_RESOURCE_FLAG_NONE,
            );
            crate::descriptor::create_srv(
                std::ptr::null(),
                tex.0,
                std::ptr::null(),
                slot_handle(i),
            );
            textures.push(tex);
        }

        let host_heap = heap_ref.host().expect("host heap created");
        let snapshot = |srvs: bool, uavs: bool| {
            let (g, s, u) = host_heap.live_textures(srvs, uavs);
            (
                g,
                s.iter().map(|(h, _)| h.get() as usize).collect::<Vec<_>>(),
                u.iter().map(|h| h.get() as usize).collect::<Vec<_>>(),
            )
        };
        let (gen0, srvs, uavs) = snapshot(true, true);
        assert_eq!(srvs.len(), COUNT as usize, "all 5000 SRVs tracked");
        assert!(uavs.is_empty());

        // Overwrite slot 10 with a CBV: texture 10 is removed cleanly.
        crate::descriptor::create_cbv(std::ptr::null(), &[0x1000, 0, 256, 0], slot_handle(10));
        let (gen1, srvs, _) = snapshot(true, false);
        assert!(gen1 > gen0);
        assert_eq!(srvs.len(), (COUNT - 1) as usize);
        assert!(!srvs.contains(&(textures[10].0 as usize)));

        // Duplicate texture 0 in slot COUNT: refcount becomes 2, overwriting one
        // slot keeps texture 0 live until the other is overwritten or forgotten.
        let tex0_addr = textures[0].0 as usize;
        crate::descriptor::create_srv(
            std::ptr::null(),
            textures[0].0,
            std::ptr::null(),
            slot_handle(COUNT),
        );
        let (gen2, srvs, _) = snapshot(true, false);
        assert!(gen2 > gen1);
        assert_eq!(srvs.len(), (COUNT - 1) as usize);
        crate::descriptor::create_cbv(std::ptr::null(), &[0x2000, 0, 256, 0], slot_handle(0));
        let (gen3, srvs, _) = snapshot(true, false);
        assert!(gen3 > gen2);
        assert!(
            srvs.contains(&tex0_addr),
            "duplicate slot keeps refcount > 0"
        );
        drop(textures.swap_remove(0));
        let (gen4, srvs, _) = snapshot(true, false);
        assert!(gen4 > gen3);
        assert_eq!(srvs.len(), (COUNT - 2) as usize);
        assert!(!srvs.contains(&tex0_addr));

        // Duplicate texture 1 across two slots and forget directly while refcount == 2.
        let tex1_addr = textures[1].0 as usize;
        crate::descriptor::create_srv(
            std::ptr::null(),
            textures[1].0,
            std::ptr::null(),
            slot_handle(COUNT + 2),
        );
        drop(textures.swap_remove(1));
        let (_, srvs, _) = snapshot(true, false);
        assert!(!srvs.contains(&tex1_addr));

        // UAV tracking + SRV->UAV overwrite on one slot.
        let uav_tex = make_texture(
            D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            4,
            4,
            1,
            DXGI_FORMAT_R8G8B8A8_UNORM,
            1,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        );
        let uav_addr = uav_tex.0 as usize;
        crate::descriptor::create_srv(
            std::ptr::null(),
            uav_tex.0,
            std::ptr::null(),
            slot_handle(COUNT + 1),
        );
        assert!(snapshot(true, false).1.contains(&uav_addr));
        crate::descriptor::create_uav(
            std::ptr::null(),
            uav_tex.0,
            std::ptr::null(),
            std::ptr::null(),
            slot_handle(COUNT + 1),
        );
        let (_, srvs, uavs) = snapshot(true, true);
        assert!(!srvs.contains(&uav_addr));
        assert!(uavs.contains(&uav_addr));

        // Holding a live_textures snapshot pins the resource across title release.
        let (_, _, pinned_uavs) = host_heap.live_textures(false, true);
        drop(uav_tex);
        assert!(
            crate::resource::host_resource(uav_addr as crate::object::This).is_some(),
            "pinned snapshot must keep released resource alive until dropped"
        );
        drop(pinned_uavs);
        assert!(!snapshot(false, true).2.contains(&uav_addr));
        drop(textures);
        let (_, srvs, uavs) = snapshot(true, true);
        assert!(srvs.is_empty());
        assert!(uavs.is_empty());

        // `copy_simple` into a shader-visible heap mirrors and tracks the texture.
        let copy_tex = make_texture(
            D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            4,
            4,
            1,
            DXGI_FORMAT_R8G8B8A8_UNORM,
            1,
            D3D12_RESOURCE_FLAG_NONE,
        );
        let copy_addr = copy_tex.0 as usize;
        let mut cpu_slots = [0u64; 2 * 4];
        let src_srv = cpu_slots.as_mut_ptr() as u64;
        let src_empty = src_srv + 32;
        crate::descriptor::create_srv(std::ptr::null(), copy_tex.0, std::ptr::null(), src_srv);
        crate::descriptor::copy_simple(std::ptr::null(), 1, slot_handle(5), src_srv, 0);
        assert_eq!(snapshot(true, false).1, vec![copy_addr]);
        crate::descriptor::copy_simple(std::ptr::null(), 1, slot_handle(5), src_empty, 0);
        assert!(snapshot(true, false).1.is_empty());

        assert_eq!(host.log_messages(), None);
    }

    #[test]
    fn direct_heap_byte_copies_are_mirrored_and_tracked() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        const MEM_WRITE_WATCH: u32 = 0x0020_0000;

        let Some(host) = host() else {
            return; // no host GPU
        };
        let tex_a = make_texture(
            D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            4,
            4,
            1,
            DXGI_FORMAT_R8G8B8A8_UNORM,
            1,
            D3D12_RESOURCE_FLAG_NONE,
        );
        let tex_b = make_texture(
            D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            4,
            4,
            1,
            DXGI_FORMAT_R8G8B8A8_UNORM,
            1,
            D3D12_RESOURCE_FLAG_NONE,
        );
        let addr_a = tex_a.0 as usize;
        let addr_b = tex_b.0 as usize;

        let mut cpu_staging = [0u64; 2 * 4];
        let src_a = cpu_staging.as_mut_ptr() as u64;
        let src_b = src_a + 32;
        crate::descriptor::create_srv(std::ptr::null(), tex_a.0, std::ptr::null(), src_a);
        crate::descriptor::create_srv(std::ptr::null(), tex_b.0, std::ptr::null(), src_b);

        // 1. Plain (non-write-watched) shader-visible heap: pre-init direct copy,
        //    post-init direct copy, overwrite, and zero-clear.
        let plain_heap = make_view_heap(16);
        let plain = crate::descriptor::heap(plain_heap.0);
        let plain_slot = |i: u32| plain.start + u64::from(i) * 32;

        // Direct byte copy BEFORE `Heap::host()` exists.
        // SAFETY: both pointers reference live 32-byte descriptor slots.
        unsafe { std::ptr::copy_nonoverlapping(src_a as *const u8, plain_slot(2) as *mut u8, 32) };
        let plain_host = plain.host().expect("plain host view heap");
        assert!(plain_host.live_texture_addrs(true, false).1.is_empty());
        assert_eq!(plain.sync_direct_copies(true), 1);
        assert_eq!(plain_host.live_texture_addrs(true, false).1, vec![addr_a]);
        assert_eq!(plain.sync_direct_copies(true), 0);

        // Overwrite slot 2 directly with `tex_b`, then clear it to zeroes.
        // SAFETY: both pointers reference live 32-byte descriptor slots.
        unsafe { std::ptr::copy_nonoverlapping(src_b as *const u8, plain_slot(2) as *mut u8, 32) };
        assert_eq!(plain.sync_direct_copies(true), 1);
        assert_eq!(plain_host.live_texture_addrs(true, false).1, vec![addr_b]);

        // SAFETY: slot 2 lies inside the live 16-slot view heap.
        unsafe { std::ptr::write_bytes(plain_slot(2) as *mut u8, 0, 32) };
        assert_eq!(plain.sync_direct_copies(true), 1);
        assert!(plain_host.live_texture_addrs(true, false).1.is_empty());

        // 2. Write-watched placed shader-visible heap spanning two 64 KiB pages
        //    (4096 slots * 32 bytes = 128 KiB).
        struct WatchedMemory(*mut u8);
        impl Drop for WatchedMemory {
            fn drop(&mut self) {
                if !self.0.is_null() {
                    // SAFETY: allocated by `VirtualAlloc` below.
                    unsafe { VirtualFree(self.0.cast(), 0, MEM_RELEASE) };
                }
            }
        }
        const SLOTS: u32 = 4096;
        const BYTES: usize = SLOTS as usize * 32;
        // SAFETY: requests a fresh zeroed 128 KiB write-watched region.
        let mem = WatchedMemory(unsafe {
            VirtualAlloc(
                std::ptr::null(),
                BYTES,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
            .cast()
        });
        assert!(!mem.0.is_null(), "VirtualAlloc(MEM_WRITE_WATCH) failed");

        #[repr(C)]
        struct PlacedDesc {
            desc: [u32; 4],
            placement: [u64; 2],
        }
        let placed = PlacedDesc {
            desc: [0, SLOTS, 1, 0],
            placement: [mem.0 as u64, BYTES as u64],
        };
        let mut raw_heap: crate::object::This = std::ptr::null();
        assert_eq!(
            crate::descriptor::create_heap(
                std::ptr::null(),
                std::ptr::from_ref(&placed).cast(),
                &crate::object::guid(
                    0x8efb471d,
                    0x616c,
                    0x4f49,
                    [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
                ),
                &mut raw_heap,
            ),
            crate::object::S_OK
        );
        let watched_heap = OwnedObject(raw_heap);
        let watched = crate::descriptor::heap(watched_heap.0);
        assert_eq!(watched.start, mem.0 as u64);
        let watched_slot = |i: u32| watched.start + u64::from(i) * 32;

        // Copy `tex_a` directly into slot 5 (page 0) before `host()` is created.
        // SAFETY: slot 5 lies within the 4096-slot placed heap.
        unsafe {
            std::ptr::copy_nonoverlapping(src_a as *const u8, watched_slot(5) as *mut u8, 32)
        };
        let watched_host = watched.host().expect("watched host view heap");
        assert_eq!(watched.sync_direct_copies(true), 1);
        assert_eq!(watched_host.live_texture_addrs(true, false).1, vec![addr_a]);
        // Clean sync scans zero dirty pages.
        assert_eq!(watched.sync_direct_copies(true), 0);

        // Direct copy into slot 2050 (second 64 KiB page) is picked up by `sync_all_heaps`.
        // SAFETY: slot 2050 lies within the 4096-slot placed heap.
        unsafe {
            std::ptr::copy_nonoverlapping(src_b as *const u8, watched_slot(2050) as *mut u8, 32);
        }
        crate::descriptor::sync_all_heaps();
        let mut live = watched_host.live_texture_addrs(true, false).1;
        live.sort_unstable();
        let mut expected = vec![addr_a, addr_b];
        expected.sort_unstable();
        assert_eq!(live, expected);
        assert_eq!(watched.sync_direct_copies(true), 0);

        assert_eq!(host.log_messages(), None);
    }

    #[test]
    fn bindless_tracker_forget_is_aba_safe_when_pointer_reused() {
        let Some(heap) = HostHeap::new(false, 16) else {
            return; // no host GPU
        };
        let reused_addr = 0x1234_5678usize;
        // Resource A mirrored at slot 5, then released without rewriting slot 5.
        heap.track_slot(5, Some((reused_addr, false, 8)));
        heap.track_slot(6, Some((reused_addr, false, 3)));
        let first = || heap.tex_tracker.lock().unwrap().first_mip(reused_addr);
        assert_eq!(first(), 3);
        heap.track_slot(6, Some((reused_addr, false, 5)));
        assert_eq!(first(), 5);
        heap.track_slot(6, None);
        assert_eq!(first(), 8);
        assert_eq!(heap.live_texture_addrs(true, false).1, vec![reused_addr]);
        heap.forget_resource(reused_addr);
        assert!(heap.live_texture_addrs(true, false).1.is_empty());
        // New resource B allocated at the same pointer and mirrored into slot 7.
        heap.track_slot(7, Some((reused_addr, false, 0)));
        // Later, stale slot 5 is overwritten with null/another view: slot 7 stays live!
        heap.track_slot(5, None);
        assert_eq!(heap.live_texture_addrs(true, false).1, vec![reused_addr]);
        // And UAV slots share the same ABA guarantee.
        heap.track_slot(2, Some((reused_addr, true, 0)));
        heap.forget_resource(reused_addr);
        heap.track_slot(9, Some((reused_addr, true, 0)));
        heap.track_slot(2, None);
        assert_eq!(heap.live_texture_addrs(false, true).2, vec![reused_addr]);
    }

    #[test]
    fn descriptor_forget_clears_host_tracker_without_by_resource_entry() {
        let Some(_host) = host() else {
            return; // no host GPU
        };
        let heap = make_view_heap(16);
        let host_heap = crate::descriptor::heap(heap.0)
            .host()
            .expect("host view heap");
        // Simulate a `View::Placed` texture tracked in `HostHeap::tex_tracker`
        // that never had an entry in `descriptor::VIEWS.by_resource`.
        let placed_addr = 0xfeed_face_0000usize;
        host_heap.track_slot(3, Some((placed_addr, false, 0)));
        assert_eq!(
            host_heap.live_texture_addrs(true, false).1,
            vec![placed_addr]
        );
        crate::descriptor::forget(placed_addr);
        assert!(
            host_heap.live_texture_addrs(true, false).1.is_empty(),
            "descriptor::forget must evict host tracker entries even when by_resource is absent"
        );
    }
}

#[cfg(test)]
mod structured_alias_tests {
    use super::*;
    use crate::descriptor::{create_placed_srv, view};

    fn raw_srv_desc(first_dword: u64, dwords: u32) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
        D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: first_dword,
                    NumElements: dwords,
                    StructureByteStride: 0,
                    Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                },
            },
        }
    }

    #[test]
    fn structured_stride_requires_a_view_of_the_matching_access_class() {
        let mut srv = [0; 10];
        srv[1] = D3D12_SRV_DIMENSION_BUFFER.0 as u32;
        srv[2] = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv[6] = 2; // NumElements
        srv[7] = 16; // StructureByteStride
        assert_eq!(
            structured_stride(&View::Shader(0, Some(srv)), false),
            Some(16)
        );
        assert_eq!(structured_stride(&View::Shader(0, Some(srv)), true), None);

        let mut uav = [0; 10];
        uav[1] = D3D12_UAV_DIMENSION_BUFFER.0 as u32;
        uav[4] = 2; // NumElements
        uav[5] = 32; // StructureByteStride
        assert_eq!(
            structured_stride(&View::Unordered(0, 0, Some(uav)), true),
            Some(32)
        );
        assert_eq!(
            structured_stride(&View::Unordered(0, 0, Some(uav)), false),
            None
        );

        uav[5] = 6;
        assert_eq!(
            structured_stride(&View::Unordered(0, 0, Some(uav)), true),
            None
        );
        srv[8] = D3D12_BUFFER_SRV_FLAG_RAW.0 as u32;
        assert_eq!(
            structured_stride(&View::Shader(0, Some(srv)), false),
            Some(0)
        );
    }

    #[test]
    fn structured_stride_recognizes_a_typeless_raw_srv() {
        let mut words = [0; 10];
        words[0] = DXGI_FORMAT_R32_TYPELESS.0 as u32;
        words[1] = D3D12_SRV_DIMENSION_BUFFER.0 as u32;
        words[2] = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        words[6] = 16; // NumElements
        words[8] = D3D12_BUFFER_SRV_FLAG_RAW.0 as u32;
        assert_eq!(
            structured_stride(&View::Shader(0, Some(words)), false),
            Some(0)
        );

        words[7] = 16; // StructureByteStride
        assert_eq!(
            structured_stride(&View::Shader(0, Some(words)), false),
            None
        );
        words[7] = 0;

        words[0] = DXGI_FORMAT_R32_UINT.0 as u32;
        assert_eq!(
            structured_stride(&View::Shader(0, Some(words)), false),
            None
        );

        words[0] = DXGI_FORMAT_R32_TYPELESS.0 as u32;
        words[8] = D3D12_BUFFER_SRV_FLAG_NONE.0 as u32;
        assert_eq!(
            structured_stride(&View::Shader(0, Some(words)), false),
            None
        );
    }

    #[test]
    fn structured_stride_recognizes_a_typeless_raw_uav() {
        let mut words = [0; 10];
        words[0] = DXGI_FORMAT_R32_TYPELESS.0 as u32;
        words[1] = D3D12_UAV_DIMENSION_BUFFER.0 as u32;
        words[4] = 16; // NumElements
        words[8] = D3D12_BUFFER_UAV_FLAG_RAW.0 as u32;
        assert_eq!(
            structured_stride(&View::Unordered(0, 0, Some(words)), true),
            Some(0)
        );

        words[5] = 16; // StructureByteStride
        assert_eq!(
            structured_stride(&View::Unordered(0, 0, Some(words)), true),
            None
        );
        words[5] = 0;

        words[0] = DXGI_FORMAT_R32_UINT.0 as u32;
        assert_eq!(
            structured_stride(&View::Unordered(0, 0, Some(words)), true),
            None
        );

        words[0] = DXGI_FORMAT_R32_TYPELESS.0 as u32;
        words[8] = D3D12_BUFFER_UAV_FLAG_NONE.0 as u32;
        assert_eq!(
            structured_stride(&View::Unordered(0, 0, Some(words)), true),
            None
        );
    }

    fn placed_raw_srv_handle(
        slot: &mut [u64; 16],
        address: u64,
        len: u64,
        first_dword: u32,
        dwords: u32,
    ) -> u64 {
        let mut resource = [0u32; 14];
        resource[0] = 1; // BUFFER
        resource[4] = len as u32;
        resource[5] = (len >> 32) as u32;
        let srv = raw_srv_desc(u64::from(first_dword), dwords);
        // SAFETY: `D3D12_SHADER_RESOURCE_VIEW_DESC` is 40 bytes (`[u32; 10]`).
        let srv_words: [u32; 10] = unsafe {
            std::ptr::from_ref(&srv)
                .cast::<[u32; 10]>()
                .read_unaligned()
        };
        let mut desc = [0u32; 16];
        desc[..10].copy_from_slice(&srv_words);
        desc[10] = address as u32;
        desc[11] = (address >> 32) as u32;
        let handle = slot.as_mut_ptr() as u64;
        create_placed_srv(std::ptr::null(), &resource, &desc, handle);
        handle
    }

    struct ShadowGuard {
        resource: usize,
        address: u64,
    }

    impl Drop for ShadowGuard {
        fn drop(&mut self) {
            crate::shadow::unregister(self.resource, self.address);
        }
    }

    #[test]
    fn structured_alias_preserves_exact_byte_bounds() {
        let alias = structured_alias(raw_srv_desc(16, 32), 64).expect("64-byte aligned raw SRV");
        assert_eq!(alias.Format, DXGI_FORMAT_UNKNOWN);
        assert_eq!(alias.ViewDimension, D3D12_SRV_DIMENSION_BUFFER);
        // SAFETY: `ViewDimension == D3D12_SRV_DIMENSION_BUFFER`.
        let buf = unsafe { alias.Anonymous.Buffer };
        assert_eq!(buf.FirstElement, 1);
        assert_eq!(buf.NumElements, 2);
        assert_eq!(buf.StructureByteStride, 64);
        assert_eq!(buf.Flags, D3D12_BUFFER_SRV_FLAG_NONE);
    }

    #[test]
    fn structured_alias_rejects_fractional_typed_structured_and_non_buffer_views() {
        // Fractional byte offset (4 bytes % 64 != 0).
        assert!(structured_alias(raw_srv_desc(1, 32), 64).is_none());
        // Fractional byte length (124 bytes % 64 != 0).
        assert!(structured_alias(raw_srv_desc(16, 31), 64).is_none());
        // Zero elements.
        assert!(structured_alias(raw_srv_desc(0, 0), 64).is_none());
        // Already-structured SRV must not be reinterpreted.
        let mut structured = raw_srv_desc(0, 2);
        structured.Format = DXGI_FORMAT_UNKNOWN;
        structured.Anonymous.Buffer.StructureByteStride = 64;
        structured.Anonymous.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
        assert!(structured_alias(structured, 64).is_none());
        // Typed non-raw buffer SRV must not be reinterpreted.
        let typed = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: 0,
                    NumElements: 8,
                    StructureByteStride: 0,
                    Flags: D3D12_BUFFER_SRV_FLAG_NONE,
                },
            },
        };
        assert!(structured_alias(typed, 64).is_none());
        // Non-buffer (texture) SRV must not be aliased.
        let tex = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            ..Default::default()
        };
        assert!(structured_alias(tex, 64).is_none());
        // Invalid strides.
        assert!(structured_alias(raw_srv_desc(0, 16), 0).is_none());
        assert!(structured_alias(raw_srv_desc(0, 16), 6).is_none());
        assert!(structured_alias(raw_srv_desc(0, 16), 4096).is_none());
    }

    #[test]
    fn make_structured_at_rejects_range_past_placed_buffer() {
        let Some(_) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let backing = vec![0u8; 1024];
        let base = backing.as_ptr() as u64;
        let _guard = ShadowGuard {
            resource: base as usize,
            address: base,
        };
        crate::shadow::register(base as usize, base, backing.len() as u64);

        // Placed length is 128 bytes, but view reads FirstElement=16 (64B) + 128B = 192B > 128B.
        let mut slot = [0u64; 16];
        let handle = placed_raw_srv_handle(&mut slot, base + 64, 128, 16, 32);
        let v = view(handle).expect("placed raw SRV");
        let dyn_idx = heap.dynamic(1).expect("dynamic slot");
        assert!(!heap.make_structured_at(dyn_idx, &v, 64));
    }

    /// Buffer views mirrored for draws are tracked and refreshed when a
    /// ResourceDescriptorHeap path asks; deferred (non-draw) mirrors are not.
    #[test]
    fn bindless_buffer_views_are_tracked_and_refreshed() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let mut backing = vec![0u8; 4096];
        backing[..16].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        let base = backing.as_ptr() as u64;
        let _guard = ShadowGuard {
            resource: base as usize,
            address: base,
        };
        crate::shadow::register(base as usize, base, backing.len() as u64);

        let mut slot = [0u64; 16];
        // 64 dwords from element 0: a 256-byte raw view of the buffer start.
        let handle = placed_raw_srv_handle(&mut slot, base, 256, 0, 64);

        assert!(heap.mirror(0, handle, false).is_some());
        assert_eq!(
            heap.live_buffer_count(),
            0,
            "deferred buffer mirrors must not track yet"
        );

        assert!(heap.mirror(0, handle, true).is_some());
        assert_eq!(
            heap.live_buffer_count(),
            1,
            "draw-time mirror must track the buffer range"
        );

        backing[..16].fill(0xAB);
        let list = host.new_list().expect("host list");
        heap.refresh_bindless_buffers(&list);
        list.close();
        host.execute(&[&list]);

        let shadow = crate::shadow::find(base, 16).expect("shadow");
        assert_eq!(
            shadow.bytes(base, 16).expect("mapped"),
            &[0xAB; 16],
            "bindless refresh must copy title writes into the host copy"
        );

        heap.make_null(0);
        assert_eq!(heap.live_buffer_count(), 0);
    }

    #[test]
    fn structured_alias_reads_exact_records_with_sentinel_and_mutation_checks() {
        use windows::Win32::Foundation::CloseHandle;
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
        use windows::core::{Interface, s};

        let Some(host) = host() else {
            return; // no host GPU
        };
        const SOURCE: &str = r#"
#define RS "DescriptorTable(SRV(t0, numDescriptors=1)), DescriptorTable(SRV(t1, numDescriptors=1)), UAV(u0)"
struct Record { uint w[16]; };
StructuredBuffer<Record> aliased : register(t0);
ByteAddressBuffer canonical_raw : register(t1);
RWByteAddressBuffer dst : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    uint count = 0;
    uint stride = 0;
    aliased.GetDimensions(count, stride);
    dst.Store(0, count);
    dst.Store(4, stride);
    for (uint r = 0; r < 2; ++r) {
        Record rec = aliased[r];
        for (uint i = 0; i < 16; ++i) {
            dst.Store(8 + (r * 16 + i) * 4, rec.w[i]);
        }
    }
    dst.Store(136, canonical_raw.Load(0));
}
"#;
        let mut blob: Option<ID3DBlob> = None;
        let mut errors: Option<ID3DBlob> = None;
        // SAFETY: source string and entry names are static and valid.
        unsafe {
            D3DCompile(
                SOURCE.as_ptr().cast(),
                SOURCE.len(),
                s!("structured_alias_readback"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut blob,
                Some(&mut errors),
            )
            .unwrap_or_else(|e| {
                let msg = errors
                    .as_ref()
                    .map(|b| {
                        let p = b.GetBufferPointer().cast::<u8>();
                        String::from_utf8_lossy(std::slice::from_raw_parts(p, b.GetBufferSize()))
                            .into_owned()
                    })
                    .unwrap_or_default();
                panic!("D3DCompile failed: {e}: {msg}");
            });
        }
        let blob = blob.unwrap();
        // SAFETY: `blob` owns compiled DXBC bytes.
        let code = unsafe {
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
        };
        // SAFETY: `code` contains embedded root signature and compute shader.
        let (root, pso): (ID3D12RootSignature, ID3D12PipelineState) = unsafe {
            let root = host.device.CreateRootSignature(0, code).unwrap();
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: code.as_ptr().cast(),
                    BytecodeLength: code.len(),
                },
                ..Default::default()
            };
            let pso = host.device.CreateComputePipelineState(&desc).unwrap();
            (root, pso)
        };

        let make_buffer = |kind: D3D12_HEAP_TYPE, flags, state| -> ID3D12Resource {
            let mut b: Option<ID3D12Resource> = None;
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 512,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: flags,
                ..Default::default()
            };
            let heap = D3D12_HEAP_PROPERTIES {
                Type: kind,
                ..Default::default()
            };
            // SAFETY: valid buffer creation parameters.
            unsafe {
                host.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        state,
                        None,
                        &mut b,
                    )
                    .unwrap();
            }
            b.unwrap()
        };

        let out = make_buffer(
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        );
        let readback = make_buffer(
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );

        let mut backing = vec![0xCCu8; 4096];
        let base = backing.as_ptr() as u64;
        let _guard = ShadowGuard {
            resource: base as usize,
            address: base,
        };
        crate::shadow::register(base as usize, base, backing.len() as u64);

        let write_word = |buf: &mut [u8], byte_off: usize, val: u32| {
            buf[byte_off..byte_off + 4].copy_from_slice(&val.to_le_bytes());
        };
        let fill_sentinels = |buf: &mut [u8], start: usize, len: usize, pat: u8| {
            buf[start..start + len].fill(pat);
        };

        // Case 1: 64-byte aligned start inside shadow (`shadow.offset == 64`).
        // Case 2: 16-byte aligned (raw-valid) but unaligned for stride 64 (`shadow.offset == 528`).
        for (tag, placed_off) in [(0x1000_0000u32, 64usize), (0x2000_0000u32, 528usize)] {
            // 64B prefix sentinel + 128B (2 records) + 64B suffix sentinel = 256B.
            let prefix_off = placed_off;
            let rec_off = placed_off + 64;
            let suffix_off = placed_off + 192;
            fill_sentinels(&mut backing, prefix_off, 64, 0xA5);
            for i in 0..32u32 {
                write_word(&mut backing, rec_off + i as usize * 4, tag | i);
            }
            fill_sentinels(&mut backing, suffix_off, 64, 0x5A);

            let mut slot = [0u64; 16];
            let handle = placed_raw_srv_handle(&mut slot, base + placed_off as u64, 192, 16, 32);
            let heap = HostHeap::new(false, 4).expect("host heap");
            let v = heap
                .mirror(0, handle, true)
                .expect("canonical slot 0 raw mirror");
            let dyn_idx = heap.dynamic(1).expect("dynamic alias slot");
            assert!(
                heap.make_structured_at(dyn_idx, &v, 64),
                "make_structured_at failed for offset {placed_off}"
            );

            // Mutate dword 17 after creating the alias to prove submission-time
            // refresh updates the backing resource (buffer shadow or the view's
            // own aligned copy).
            let mutated_word = tag | 0x00FF_0011;
            write_word(&mut backing, rec_off + 17 * 4, mutated_word);

            let list = host.new_list().expect("host list");
            assert!(heap.refresh_slot(dyn_idx, &list));
            if placed_off % 64 != 0 {
                // `dyn_idx` reads the view's own aligned copy rather than the
                // buffer shadow, so refresh the canonical raw slot's first
                // record word independently for its readback check.
                crate::shadow::find(base + rec_off as u64, 4)
                    .expect("registered shadow")
                    .refresh(&list, base + rec_off as u64, 4);
            }
            // SAFETY: open list with live compute root signature, PSO, and heaps.
            unsafe {
                let l = &list.list;
                l.SetComputeRootSignature(&root);
                l.SetPipelineState(&pso);
                l.SetDescriptorHeaps(&[Some(heap.heap.clone())]);
                l.SetComputeRootDescriptorTable(0, heap.gpu(dyn_idx));
                l.SetComputeRootDescriptorTable(1, heap.gpu(0));
                l.SetComputeRootUnorderedAccessView(2, out.GetGPUVirtualAddress());
                l.Dispatch(1, 1, 1);
                l.ResourceBarrier(&[crate::host::transition(
                    &out,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                l.CopyBufferRegion(&readback, 0, &out, 0, 140);
                l.ResourceBarrier(&[crate::host::transition(
                    &out,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                )]);
            }
            list.close();
            host.execute(&[&list]);

            // Validate descriptor binding cleanly through a standalone queue when
            // the D3D12 InfoQueue is active (`VICTORY_D3D12_DEBUG=1`) so we inspect
            // messages before `host.execute` clears them.
            if let Ok(iq) = host.device.cast::<ID3D12InfoQueue>() {
                // SAFETY: standalone queue/allocator/list/fence on `host.device`.
                unsafe {
                    let queue: ID3D12CommandQueue = host
                        .device
                        .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                            ..Default::default()
                        })
                        .unwrap();
                    let alloc: ID3D12CommandAllocator = host
                        .device
                        .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                        .unwrap();
                    let cl: ID3D12GraphicsCommandList = host
                        .device
                        .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
                        .unwrap();
                    cl.SetComputeRootSignature(&root);
                    cl.SetPipelineState(&pso);
                    cl.SetDescriptorHeaps(&[Some(heap.heap.clone())]);
                    cl.SetComputeRootDescriptorTable(0, heap.gpu(dyn_idx));
                    cl.SetComputeRootDescriptorTable(1, heap.gpu(0));
                    cl.SetComputeRootUnorderedAccessView(2, out.GetGPUVirtualAddress());
                    cl.Dispatch(1, 1, 1);
                    cl.Close().unwrap();

                    let before = iq.GetNumStoredMessages();
                    queue.ExecuteCommandLists(&[Some(cl.into())]);
                    let fence: ID3D12Fence =
                        host.device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
                    let event = CreateEventW(None, false, false, None).unwrap();
                    queue.Signal(&fence, 1).unwrap();
                    fence.SetEventOnCompletion(1, event).unwrap();
                    WaitForSingleObject(event, INFINITE);
                    let _ = CloseHandle(event);

                    let after = iq.GetNumStoredMessages();
                    for idx in before.min(after)..after {
                        let mut size = 0usize;
                        if iq.GetMessage(idx, None, &mut size).is_err() || size == 0 {
                            continue;
                        }
                        let mut raw = vec![0u64; size.div_ceil(8)];
                        let msg = raw.as_mut_ptr().cast::<D3D12_MESSAGE>();
                        if iq.GetMessage(idx, Some(msg), &mut size).is_ok() {
                            let m = &*msg;
                            let desc =
                                std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength);
                            let text = String::from_utf8_lossy(desc);
                            assert_ne!(m.ID.0, 1387, "unexpected ID 1387: {text}");
                            assert!(
                                m.Severity.0 > D3D12_MESSAGE_SEVERITY_ERROR.0,
                                "unexpected D3D12 error {}: {text}",
                                m.ID.0
                            );
                        }
                    }
                }
            }

            let mut mapped = std::ptr::null_mut();
            let mut dwords = [0u32; 35];
            // SAFETY: execution completed and readback holds 140 bytes (35 dwords).
            unsafe {
                let range = D3D12_RANGE { Begin: 0, End: 140 };
                readback.Map(0, Some(&range), Some(&mut mapped)).unwrap();
                std::ptr::copy_nonoverlapping(mapped.cast::<u32>(), dwords.as_mut_ptr(), 35);
                readback.Unmap(0, None);
            }
            assert_eq!(dwords[0], 2, "GetDimensions count for offset {placed_off}");
            assert_eq!(
                dwords[1], 64,
                "GetDimensions stride for offset {placed_off}"
            );
            for i in 0..32u32 {
                let expected = if i == 17 { mutated_word } else { tag | i };
                assert_eq!(
                    dwords[2 + i as usize],
                    expected,
                    "record dword {i} mismatch for offset {placed_off}"
                );
            }
            assert_eq!(
                dwords[34], tag,
                "canonical slot 0 must stay a valid raw SRV for offset {placed_off}"
            );
            assert!(
                backing[prefix_off..prefix_off + 64]
                    .iter()
                    .all(|&b| b == 0xA5)
            );
            assert!(
                backing[suffix_off..suffix_off + 64]
                    .iter()
                    .all(|&b| b == 0x5A)
            );
        }

        // SAFETY: plain device health query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    #[test]
    fn descriptor_table_over_128_mib_loose_budget_keeps_first_shadow_for_refresh() {
        let _retire = crate::host::isolated_retire_scope();
        let _budget = crate::shadow::isolated_loose_budget(128 << 20);
        let Some(host) = host() else {
            return;
        };
        let heap = HostHeap::root_views().expect("root view heap");
        let count = 33u32;
        let first = heap.dynamic(count).expect("33 dynamic slots");

        #[repr(align(16))]
        struct Aligned([u32; 16]);
        let mut backing = Aligned([0x1111_1111; 16]);
        let addr0 = (&raw const backing.0) as u64;

        let mut slot_mem = [0u64; 16];
        let h0 = placed_raw_srv_handle(&mut slot_mem, addr0, 64, 0, 16);
        let view0 = crate::descriptor::view(h0).expect("placed raw srv 0");
        assert!(heap.make_at(first, &view0), "slot 0 raw srv must succeed");

        let s0_initial = crate::shadow::find(addr0, 64).expect("loose shadow 0");
        let weak0 = Arc::downgrade(&s0_initial);
        let initial_gpu = s0_initial.gpu(addr0);
        drop(s0_initial);
        // `LOOSE` holds 1 strong ref and `heap.kept[first]` holds 1 strong ref.
        assert_eq!(weak0.strong_count(), 2);

        // Populate 32 more distinct 4 MiB loose windows in the same table
        // (33 * 4 MiB = 132 MiB > 128 MiB budget). Because every live slot
        // pins its `Arc<Shadow>`, slot 0's shadow must not be evicted.
        let synthetic_base = 0x2000_0000_0000_u64;
        let mut weak1 = None;
        for i in 1..count {
            let addr = synthetic_base + u64::from(i) * (4 << 20);
            let h = placed_raw_srv_handle(&mut slot_mem, addr, 64, 0, 16);
            let view = crate::descriptor::view(h).expect("placed raw srv");
            assert!(
                heap.make_at(first + i, &view),
                "slot {i} raw srv must succeed"
            );
            if i == 1 {
                weak1 = Some(Arc::downgrade(
                    &crate::shadow::find(addr, 64).expect("loose shadow 1"),
                ));
            }
        }
        let weak1 = weak1.expect("window 1 weak ref");

        // Later refresh lookup must resolve the exact same `Arc<Shadow>` whose
        // buffer slot 0 points to, not a newly allocated replacement.
        let s0_after = crate::shadow::find(addr0, 64).expect("loose shadow 0 still cached");
        assert_eq!(
            s0_after.gpu(addr0),
            initial_gpu,
            "slot 0 descriptor must still name the same shadow buffer after >32 windows"
        );
        assert!(
            weak0
                .upgrade()
                .is_some_and(|orig| Arc::ptr_eq(&orig, &s0_after)),
            "refresh must target the exact Arc<Shadow> retained by slot 0"
        );

        // Mutating title memory and refreshing writes into the exact shadow
        // buffer pinned by slot 0 when submitted.
        backing.0.fill(0x5A5A_5A5A);
        let list = host.new_list().expect("host list");
        s0_after.refresh(&list, addr0, 64);
        list.close();
        host.execute(&[&list]);
        list.reset();
        let copied = s0_after.read_test_bytes(addr0, 64);
        assert_eq!(copied, vec![0x5Au8; 64]);
        drop(s0_after);

        // Once the dynamic table slots are released, descriptor-free idle
        // windows become eligible for LRU eviction again.
        heap.release(first, count);
        assert_eq!(
            weak0.strong_count(),
            1,
            "releasing dynamic slots must drop HostHeap's Arc<Shadow> pin"
        );
        assert_eq!(
            weak1.strong_count(),
            1,
            "releasing dynamic slots must unpin window 1 as well"
        );
        // Window 1 is now the oldest untouched entry (window 0 was refreshed
        // above); inserting one more window drains the cache back to 32 entries
        // and evicts window 1 immediately.
        let extra_addr = synthetic_base + 40 * (4 << 20);
        let _extra = crate::shadow::find(extra_addr, 64).expect("extra loose window");
        assert_eq!(
            weak1.strong_count(),
            0,
            "idle LRU loose window must be evicted once no descriptor slot pins it"
        );
    }
}
