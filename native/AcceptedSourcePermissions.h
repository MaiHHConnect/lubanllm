#pragma once
#include <sys/mount.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>

// Read-only accepted artifacts on exFAT expose mount permissions, not POSIX file privacy.
// Scratch, RNG and newly written native state retain their separate strict 0600 contracts.
inline bool lubanAcceptedSourcePermissions(int fd, const struct stat& source) {
  if (source.st_uid != getuid()) return false;
  const auto mode = source.st_mode & 07777;
  if (mode == 0600) return true;
  struct statfs volume{};
  return mode == 0700 && fstatfs(fd, &volume) == 0 && std::strcmp(volume.f_fstypename, "exfat") == 0;
}
