//! Read native identity/capacity from the very same enumerated adapter.

#[cfg(target_os = "windows")]
mod windows;

#[derive(Default)]
pub(super) struct NativeMetadata {
    pub identity: Option<String>,
    pub vram_bytes: Option<u64>,
    pub software_or_indirect: bool}

#[cfg(any(target_os = "windows", target_os = "linux"))]
pub(super) fn metadata(adapter: &wgpu::Adapter, info: &wgpu::AdapterInfo) -> NativeMetadata {
    use ash::vk;

    if info.backend != wgpu::Backend::Vulkan {
        return NativeMetadata::default();
    }
    // SAFETY: All queries use this guard's physical device and its own live
    // Vulkan instance. We do not create, mutate, destroy, or enumerate handles.
    let Some(native) = (unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>() }) else {
        return NativeMetadata::default();
    };
    let shared = native.shared_instance();
    let instance = shared.raw_instance();
    let physical_device = native.raw_physical_device();
    let properties = native.physical_device_capabilities().properties();
    let mut metadata = NativeMetadata::default();
    if shared.instance_api_version() >= vk::API_VERSION_1_1
        && properties.api_version >= vk::API_VERSION_1_1
    {
        let mut ids = vk::PhysicalDeviceIDProperties::default();
        let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut ids);
        // SAFETY: Vulkan 1.1 supports this query and ID properties for both the
        // instance and physical device; the output structures live for the call.
        unsafe { instance.get_physical_device_properties2(physical_device, &mut properties) };
        if ids.device_uuid.iter().any(|byte| *byte != 0) {
            metadata.identity = Some(format!("uuid-{}", hex::encode(ids.device_uuid)));
        }
        #[cfg(target_os = "windows")]
        if let Some((low, high)) = windows::vulkan_luid(ids.device_luid_valid != 0, ids.device_luid)
        {
            metadata.identity = Some(windows::vulkan_identity(low, high, ids.device_node_mask));
            let adapter_type = windows::adapter_type(low, high)
                .map_err(|reason| {
                    super::diagnostic(format_args!(
                        "adapter type unavailable for {} (luid-{:08x}-{:08x}): {reason}",
                        info.name, high as u32, low
                    ));
                })
                .ok();
            // Classify this Vulkan physical device's original Windows LUID.
            // A failed query preserves its identity and capacity.
            metadata.software_or_indirect = windows::software_or_indirect(0, adapter_type);
        }
    }
    metadata.vram_bytes = if info.device_type == wgpu::DeviceType::DiscreteGpu {
        // SAFETY: This immutable query uses the same live physical device.
        let memory = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        memory
            .memory_heaps
            .iter()
            .take(memory.memory_heap_count as usize)
            .filter(|heap| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .try_fold(0_u64, |total, heap| total.checked_add(heap.size))
            .filter(|total| *total > 0)
    } else {
        // An integrated GPU's shared system memory is not dedicated VRAM.
        None
    };
    metadata
}

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
pub(super) fn metadata(_adapter: &wgpu::Adapter, _info: &wgpu::AdapterInfo) -> NativeMetadata {
    // WGPU 29's Metal Adapter does not expose its native MTLDevice/registryID.
    // Keep the stable hardware-description fallback and do not claim a working-
    // set recommendation or unified memory size is physical dedicated VRAM.
    NativeMetadata::default()
}
