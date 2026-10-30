// Shared native subprocess environment, reused from the verified runtime packager.
export function cleanRuntimeEnvironment() {
  const env: Record<string, string> = {};
  for (const [key, value] of Object.entries(process.env)) {
    if (value !== undefined && !/^(PYTHON|DYLD_)/.test(key) && !["VIRTUAL_ENV", "CONDA_PREFIX", "LUBAN_MODEL_LOCK_FD"].includes(key)) env[key] = value;
  }
  env.PYTORCH_ENABLE_MPS_FALLBACK = "0";
  return env;
}
