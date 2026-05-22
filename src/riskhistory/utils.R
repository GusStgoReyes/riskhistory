
# Auto-detect repo root: this file is at src/riskhistory/config.R
# so repo root is 2 levels up
get_repo_path <- function() {
  # First check environment variable
  env_path <- Sys.getenv("RISKHISTORY_ROOT", unset = "")
  if (env_path != "") {
    return(normalizePath(env_path))
  }
  # Auto-detect from this file's location
  # Try to get the path of the sourced script
  if (sys.nframe() > 0) {
    # When sourced, get the directory of this script
    for (i in sys.nframe():1) {
      if (!is.null(sys.frame(i)$ofile)) {
        script_dir <- dirname(normalizePath(sys.frame(i)$ofile))
        return(normalizePath(file.path(script_dir, "..", "..")))
      }
    }
  }

  # Fallback: use current working directory and look for markers
  cwd <- getwd()
  # Check if we're in the repo root
  if (file.exists(file.path(cwd, "src", "riskhistory", "config.R"))) {
    return(normalizePath(cwd))
  }
  # Check if we're in src/riskhistory
  if (basename(cwd) == "riskhistory" && basename(dirname(cwd)) == "src") {
    return(normalizePath(file.path(cwd, "..", "..")))
  }
  # Check if we're in an analysis subdirectory
  if (file.exists(file.path(cwd, "..", "config.R"))) {
    return(normalizePath(file.path(cwd, "..", "..", "..")))
  }

  stop("Could not auto-detect repo path. Set RISKHISTORY_ROOT environment variable.")
}
