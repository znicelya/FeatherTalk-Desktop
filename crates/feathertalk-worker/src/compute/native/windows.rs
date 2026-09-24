//! Classify the original Windows adapter, before Windows maps display aliases to
//! a shared render GPU. The declarations mirror the Windows SDK's d3dkmthk.h.

use std::{ffi::c_void, mem::size_of};

const DXGI_ADAPTER_FLAG_SOFTWARE: u32 = 2;
const KMTQAITYPE_ADAPTERTYPE: i32 = 15;
const SOFTWARE_DEVICE: u32 = 1 << 2;
const INDIRECT_DISPLAY_DEVICE: u32 = 1 << 6;

#[repr(C)]
struct Luid {
    low: u32,
    high: i32,
}

#[repr(C)]
struct OpenAdapterFromLuid {
    luid: Luid,
    handle: u32,
}

#[repr(C)]
struct QueryAdapterInfo {
    handle: u32,
    kind: i32,
    data: *mut c_void,
    size: u32,
}

#[repr(C)]
struct CloseAdapter {
    handle: u32,
}

#[link(name = "gdi32")]
unsafe extern "system" {
    fn D3DKMTOpenAdapterFromLuid(request: *mut OpenAdapterFromLuid) -> i32;
    fn D3DKMTQueryAdapterInfo(request: *const QueryAdapterInfo) -> i32;
    fn D3DKMTCloseAdapter(request: *const CloseAdapter) -> i32;
}

/// Vulkan exposes the Windows LUID bytes in little-endian LowPart/HighPart
/// order. An invalid or empty LUID must leave the Vulkan UUID fallback intact.
pub(super) fn vulkan_luid(valid: bool, bytes: [u8; 8]) -> Option<(u32, i32)> {
    if !valid || bytes == [0; 8] {
        return None;
    }
    Some((
        u32::from_le_bytes(bytes[..4].try_into().expect("four LUID low bytes")),
        i32::from_le_bytes(bytes[4..].try_into().expect("four LUID high bytes")),
    ))
}

/// A LUID identifies a Windows adapter, which can contain multiple GPU nodes.
pub(super) fn vulkan_identity(low: u32, high: i32, node_mask: u32) -> String {
    format!("luid-{:08x}-{:08x}-node-{node_mask:08x}", high as u32, low)
}

/// Owns only the kernel handle opened here, never the borrowed DXGI adapter.
struct KernelAdapter(u32);

impl Drop for KernelAdapter {
    fn drop(&mut self) {
        let request = CloseAdapter { handle: self.0 };
        // SAFETY: OpenAdapterFromLuid returned this nonzero handle successfully.
        // The guard has sole ownership and closes it once, including on errors.
        let _ = unsafe { D3DKMTCloseAdapter(&request) };
    }
}

pub(super) fn adapter_type(low: u32, high: i32) -> Result<u32, String> {
    let mut request = OpenAdapterFromLuid {
        luid: Luid { low, high },
        handle: 0,
    };
    // SAFETY: The input LUID and output handle have the SDK's C layout, and
    // request remains writable for the duration of the synchronous call.
    let status = unsafe { D3DKMTOpenAdapterFromLuid(&mut request) };
    if status < 0 {
        return Err(format!(
            "D3DKMTOpenAdapterFromLuid failed with NTSTATUS 0x{:08x}",
            status as u32
        ));
    }
    if request.handle == 0 {
        return Err("D3DKMTOpenAdapterFromLuid returned an empty handle".into());
    }
    let adapter = KernelAdapter(request.handle);
    let mut flags = 0_u32;
    let query = QueryAdapterInfo {
        handle: adapter.0,
        // Query the original adapter type, not ADAPTERTYPE_RENDER (57): an
        // indirect display device's renderer has the physical GPU's flags.
        kind: KMTQAITYPE_ADAPTERTYPE,
        data: (&mut flags as *mut u32).cast(),
        size: size_of::<u32>() as u32,
    };
    // SAFETY: The open handle stays live through the call. ADAPTERTYPE writes
    // exactly one 32-bit flags value to the live, correctly sized output buffer.
    let status = unsafe { D3DKMTQueryAdapterInfo(&query) };
    if status < 0 {
        return Err(format!(
            "D3DKMTQueryAdapterInfo failed with NTSTATUS 0x{:08x}",
            status as u32
        ));
    }
    Ok(flags)
}

pub(super) fn software_or_indirect(dxgi_flags: u32, adapter_type: Option<u32>) -> bool {
    dxgi_flags & DXGI_ADAPTER_FLAG_SOFTWARE != 0
        || adapter_type
            .is_some_and(|flags| flags & (SOFTWARE_DEVICE | INDIRECT_DISPLAY_DEVICE) != 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::offset_of;

    #[test]
    fn vulkan_identity_preserves_both_luid_halves() {
        let bytes = [0x6c, 0x03, 0x01, 0x00, 0x78, 0x56, 0x34, 0x92];
        assert_eq!(
            vulkan_luid(true, bytes),
            Some((0x0001_036c, 0x9234_5678_u32 as i32))
        );
        assert_eq!(vulkan_luid(false, bytes), None);
        assert_eq!(vulkan_luid(true, [0; 8]), None);
    }

    #[test]
    fn linked_gpu_nodes_with_a_shared_luid_keep_distinct_identities() {
        let first = vulkan_identity(0x0001_036c, 0x9234_5678_u32 as i32, 1);
        let second = vulkan_identity(0x0001_036c, 0x9234_5678_u32 as i32, 2);
        assert_ne!(first, second, "linked physical GPUs must remain selectable");
        assert_eq!(first, "luid-92345678-0001036c-node-00000001");
        assert_eq!(second, "luid-92345678-0001036c-node-00000002");
    }

    #[test]
    fn original_adapter_flags_exclude_display_aliases_and_software() {
        assert!(!software_or_indirect(0, Some(0x30b))); // physical RTX 2060
        assert!(software_or_indirect(0, Some(0x142))); // Oray / GameViewer IDD
        assert!(software_or_indirect(0, Some(0x105))); // Microsoft software GPU
        assert!(software_or_indirect(0, Some(SOFTWARE_DEVICE)));
        assert!(software_or_indirect(0, Some(INDIRECT_DISPLAY_DEVICE)));
        // Unknown bits cannot conceal an indirect display or software marker.
        assert!(software_or_indirect(0, Some(0x8000_0142)));
        assert!(software_or_indirect(0, Some(0x8000_0105)));
    }

    #[test]
    fn headless_non_primary_and_unknown_adapters_are_not_filtered() {
        // Neither DisplaySupported nor PostDevice is required for computation.
        for flags in [0, 1, 0x8000_0001, 0x30b] {
            assert!(!software_or_indirect(0, Some(flags)));
        }
        // A failed native query leaves the existing metadata usable. A known
        // DXGI software flag remains authoritative even when that query fails.
        assert!(!software_or_indirect(0, None));
        assert!(software_or_indirect(DXGI_ADAPTER_FLAG_SOFTWARE, None));
        assert!(software_or_indirect(
            DXGI_ADAPTER_FLAG_SOFTWARE,
            Some(0x30b)
        ));
    }

    #[test]
    fn declarations_match_the_windows_sdk_abi() {
        assert_eq!(size_of::<Luid>(), 8);
        assert_eq!(offset_of!(Luid, low), 0);
        assert_eq!(offset_of!(Luid, high), 4);
        assert_eq!(size_of::<OpenAdapterFromLuid>(), 12);
        assert_eq!(offset_of!(OpenAdapterFromLuid, handle), 8);
        assert_eq!(size_of::<CloseAdapter>(), 4);
        assert_eq!(offset_of!(QueryAdapterInfo, handle), 0);
        assert_eq!(offset_of!(QueryAdapterInfo, kind), 4);
        assert_eq!(offset_of!(QueryAdapterInfo, data), 8);
        assert_eq!(
            offset_of!(QueryAdapterInfo, size),
            8 + size_of::<*mut c_void>()
        );
        assert_eq!(
            size_of::<QueryAdapterInfo>(),
            if cfg!(target_pointer_width = "64") {
                24
            } else {
                16
            }
        );
    }
}
