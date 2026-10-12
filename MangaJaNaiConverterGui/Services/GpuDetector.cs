using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Management;
using System.Runtime.Versioning;
using System.Text.RegularExpressions;

namespace MangaJaNaiConverterGui.Services
{
    public enum BackendKind
    {
        Cuda,
        Rocm,
        DirectMl,
        Cpu,
    }

    /// <summary>
    /// Which Python backend to install. Persisted next to the Python install so that a GPU change
    /// (or an override) triggers a reinstall.
    /// </summary>
    public record BackendFlavor(BackendKind Kind, string? RocmFamily = null)
    {
        public string Id => Kind == BackendKind.Rocm ? $"rocm-{RocmFamily}" : Kind.ToString().ToLowerInvariant();

        public string Description => Kind switch
        {
            BackendKind.Cuda => "NVIDIA CUDA",
            BackendKind.Rocm => $"AMD ROCm ({RocmFamily})",
            BackendKind.DirectMl => "DirectML (ONNX Runtime)",
            _ => "CPU",
        };
    }

    public record GpuAdapter(string Name, string VendorId, string DeviceId);

    public static class GpuDetector
    {
        private const string DisplayClassKey = @"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}";

        public const string VendorAmd = "1002";
        public const string VendorNvidia = "10de";
        public const string VendorIntel = "8086";

        public const string FamilyRdna4 = "gfx120X-all";
        public const string FamilyRdna3 = "gfx110X-all";
        public const string FamilyStrixHalo = "gfx1151";

        private static readonly Regex PciIdRegex = new(@"ven_([0-9a-f]{4})&dev_([0-9a-f]{4})", RegexOptions.IgnoreCase);

        // AMD publishes Windows ROCm PyTorch wheels per GPU family at https://rocm.nightlies.amd.com/v2/<family>/.
        // Only these families currently have win_amd64 wheels.
        private static readonly Dictionary<string, string> RocmFamilyByDeviceId = new(StringComparer.OrdinalIgnoreCase)
        {
            // RDNA4 (gfx1200/gfx1201): RX 9070 / 9070 XT / 9070 GRE / AI PRO R9700, RX 9060 / 9060 XT
            ["7550"] = FamilyRdna4,
            ["7551"] = FamilyRdna4,
            ["7590"] = FamilyRdna4,
            ["7591"] = FamilyRdna4,
            // discrete RDNA3 (gfx1100-gfx1102): RX 7900 / 7800 / 7700 / 7600, PRO W7000
            ["744c"] = FamilyRdna3,
            ["7448"] = FamilyRdna3,
            ["745e"] = FamilyRdna3,
            ["747e"] = FamilyRdna3,
            ["7480"] = FamilyRdna3,
            ["7483"] = FamilyRdna3,
            ["7489"] = FamilyRdna3,
            // integrated RDNA3 (gfx1103): Radeon 780M / 760M / 740M
            ["15bf"] = FamilyRdna3,
            ["15c8"] = FamilyRdna3,
            // Strix Halo (gfx1151): Radeon 8060S / 8050S / 8040S
            ["1586"] = FamilyStrixHalo,
        };

        private static readonly HashSet<string> DiscreteRdna3DeviceIds = new(StringComparer.OrdinalIgnoreCase)
        {
            "744c", "7448", "745e", "747e", "7480", "7483", "7489",
        };

        private static readonly Regex DiscreteRdna3NameRegex = new(@"RX\s*7\d{3}|PRO W7\d{3}", RegexOptions.IgnoreCase);

        private static readonly (Regex Pattern, string Family)[] RocmFamilyByName =
        [
            (new Regex(@"RX\s*90\d0|AI PRO R9\d00", RegexOptions.IgnoreCase), FamilyRdna4),
            (new Regex(@"RX\s*7[6-9]\d0|PRO W7\d00|\b7[4-8]0M\b", RegexOptions.IgnoreCase), FamilyRdna3),
            (new Regex(@"\b80[4-6]0S\b", RegexOptions.IgnoreCase), FamilyStrixHalo),
        ];

        /// <summary>
        /// Returns the physical (PCI) display adapters that are currently present and working.
        /// Uses WMI; falls back to the display class registry key if WMI is unavailable.
        /// </summary>
        public static List<GpuAdapter> GetAdapters()
        {
            if (!OperatingSystem.IsWindows())
            {
                return [];
            }

            try
            {
                return GetAdaptersWmi();
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"GpuDetector: WMI query failed, falling back to registry: {ex.Message}");
            }

            return GetAdaptersRegistry();
        }

        [SupportedOSPlatform("windows")]
        private static List<GpuAdapter> GetAdaptersWmi()
        {
            var adapters = new List<GpuAdapter>();

            using var searcher = new ManagementObjectSearcher("SELECT Name, PNPDeviceID, Status, ConfigManagerErrorCode FROM Win32_VideoController");
            using var results = searcher.Get();

            foreach (var obj in results)
            {
                using (obj)
                {
                    var name = obj["Name"] as string ?? "";
                    var pnpId = obj["PNPDeviceID"] as string;
                    var errorCode = obj["ConfigManagerErrorCode"];

                    if (errorCode != null && Convert.ToUInt32(errorCode) != 0)
                    {
                        // disabled, driver missing/failed, or otherwise not working
                        Debug.WriteLine($"GpuDetector: skipping '{name}' (ConfigManagerErrorCode {errorCode})");
                        continue;
                    }

                    if (TryParsePciId(pnpId, out var vendorId, out var deviceId))
                    {
                        adapters.Add(new GpuAdapter(name, vendorId, deviceId));
                    }
                    else
                    {
                        // virtual adapters (Parsec, Microsoft Basic Display, remote display, ...)
                        Debug.WriteLine($"GpuDetector: skipping non-PCI adapter '{name}' ({pnpId})");
                    }
                }
            }

            return adapters;
        }

        [SupportedOSPlatform("windows")]
        private static List<GpuAdapter> GetAdaptersRegistry()
        {
            var adapters = new List<GpuAdapter>();

            try
            {
                using var classKey = Registry.LocalMachine.OpenSubKey(DisplayClassKey);
                if (classKey == null)
                {
                    return adapters;
                }

                foreach (var subKeyName in classKey.GetSubKeyNames().Where(n => n.All(char.IsDigit)))
                {
                    try
                    {
                        using var subKey = classKey.OpenSubKey(subKeyName);
                        var matchingId = subKey?.GetValue("MatchingDeviceId") as string;
                        var name = subKey?.GetValue("DriverDesc") as string;
                        if (string.IsNullOrEmpty(matchingId) || string.IsNullOrEmpty(name))
                        {
                            continue;
                        }

                        if (TryParsePciId(matchingId, out var vendorId, out var deviceId))
                        {
                            adapters.Add(new GpuAdapter(name, vendorId, deviceId));
                        }
                    }
                    catch (Exception ex)
                    {
                        Debug.WriteLine($"GpuDetector: failed to read display class subkey {subKeyName}: {ex.Message}");
                    }
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"GpuDetector: failed to read display class key: {ex.Message}");
            }

            return adapters;
        }

        private static bool TryParsePciId(string? id, out string vendorId, out string deviceId)
        {
            vendorId = deviceId = "";

            if (string.IsNullOrEmpty(id) || !id.StartsWith(@"PCI\", StringComparison.OrdinalIgnoreCase))
            {
                return false;
            }

            var match = PciIdRegex.Match(id);
            if (!match.Success)
            {
                return false;
            }

            vendorId = match.Groups[1].Value.ToLowerInvariant();
            deviceId = match.Groups[2].Value.ToLowerInvariant();
            return true;
        }

        public static string? GetRocmFamily(GpuAdapter adapter)
        {
            if (adapter.VendorId != VendorAmd)
            {
                return null;
            }

            if (RocmFamilyByDeviceId.TryGetValue(adapter.DeviceId, out var family))
            {
                return family;
            }

            return RocmFamilyByName.FirstOrDefault(x => x.Pattern.IsMatch(adapter.Name)).Family;
        }

        /// <summary>
        /// Preference order when several ROCm-capable adapters are present (higher is better):
        /// RDNA4 &gt; discrete RDNA3 &gt; Strix Halo &gt; RDNA3 iGPU.
        /// </summary>
        private static int RocmRank(GpuAdapter adapter, string family) => family switch
        {
            FamilyRdna4 => 4,
            FamilyRdna3 when DiscreteRdna3DeviceIds.Contains(adapter.DeviceId) || DiscreteRdna3NameRegex.IsMatch(adapter.Name) => 3,
            FamilyStrixHalo => 2,
            FamilyRdna3 => 1,
            _ => 0,
        };

        public static string? GetBestRocmFamily(IEnumerable<GpuAdapter> adapters) => adapters
            .Select(a => (Adapter: a, Family: GetRocmFamily(a)))
            .Where(x => x.Family != null)
            .OrderByDescending(x => RocmRank(x.Adapter, x.Family!))
            .Select(x => x.Family)
            .FirstOrDefault();

        /// <summary>
        /// Chooses the backend to install. Override with the MANGAJANAI_BACKEND environment variable
        /// (cuda, rocm, directml, cpu) and MANGAJANAI_ROCM_FAMILY (e.g. gfx110X-all).
        /// </summary>
        public static BackendFlavor DetectFlavor()
        {
            var adapters = GetAdapters();
            var overrideKind = Environment.GetEnvironmentVariable("MANGAJANAI_BACKEND")?.Trim().ToLowerInvariant();
            var overrideFamily = Environment.GetEnvironmentVariable("MANGAJANAI_ROCM_FAMILY")?.Trim();
            var detectedFamily = GetBestRocmFamily(adapters);

            foreach (var adapter in adapters)
            {
                Debug.WriteLine($"GpuDetector: found '{adapter.Name}' ven_{adapter.VendorId} dev_{adapter.DeviceId} rocm={GetRocmFamily(adapter) ?? "none"}");
            }

            switch (overrideKind)
            {
                case "cuda":
                    return new BackendFlavor(BackendKind.Cuda);
                case "rocm":
                    var family = !string.IsNullOrEmpty(overrideFamily) ? overrideFamily : detectedFamily;
                    if (family != null)
                    {
                        return new BackendFlavor(BackendKind.Rocm, family);
                    }
                    Debug.WriteLine("GpuDetector: MANGAJANAI_BACKEND=rocm but no ROCm family detected or given in MANGAJANAI_ROCM_FAMILY; using auto-detection");
                    break;
                case "directml":
                    return new BackendFlavor(BackendKind.DirectMl);
                case "cpu":
                    return new BackendFlavor(BackendKind.Cpu);
            }

            if (adapters.Count == 0 || adapters.Any(a => a.VendorId == VendorNvidia))
            {
                // Unknown hardware or NVIDIA present: keep the original prebuilt CUDA backend.
                return new BackendFlavor(BackendKind.Cuda);
            }

            if (detectedFamily != null)
            {
                return new BackendFlavor(BackendKind.Rocm, detectedFamily);
            }

            if (adapters.Any(a => a.VendorId is VendorAmd or VendorIntel))
            {
                return new BackendFlavor(BackendKind.DirectMl);
            }

            return new BackendFlavor(BackendKind.Cpu);
        }
    }
}
