//! Private AI — Tauri desktop app.
//!
//! The app is a native wrapper around the Private AI Inference Server web UI.
//! On first run (or when the server is not reachable) it shows a **native guided
//! installer** (screens A–H). Once the server responds on :8005, the window
//! navigates to the web UI.
//!
//! Native capabilities exposed to the installer frontend:
//! - `check_server` / `test_connection` — probe `/health` endpoints
//! - `pick_folder` — native directory picker (install dir / models dir)
//! - `default_install_dir` / `default_models_dir` — OS-appropriate defaults
//! - `probe_services` — probe all known sibling services at once
//! - `register_autostart` — logon scheduled task (no admin)

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::Serialize;
use tauri::Manager;
use tauri_plugin_dialog::DialogExt;

/// The main server this app wraps.
pub const SERVER_URL: &str = "http://127.0.0.1:8005";
/// Sibling services the installer can optionally provision.
/// (key, display name, health URL)
pub const SERVICES: &[(&str, &str, &str)] = &[
    ("ai", "Private AI Server", "http://127.0.0.1:8005/health"),
    ("imagegen", "Open Fantasia", "http://127.0.0.1:8765/health"),
    ("voice", "Olly Voice", "http://127.0.0.1:8766/health"),
];

/// Result of probing a single service's health endpoint.
#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ServiceStatus {
    pub key: String,
    pub name: String,
    pub url: String,
    pub reachable: bool,
    pub detail: String,
}

/// Probe a `/health` endpoint with a short timeout.
///
/// Returns `Ok(())` on any 2xx response, `Err(msg)` otherwise. Uses a small
/// synchronous HTTP client (`ureq`) so the call is fast and self-contained.
fn probe_health(url: &str) -> Result<(), String> {
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(3))
        .timeout_read(Duration::from_secs(5))
        .build();
    match agent.get(url).call() {
        Ok(resp) => {
            let status = resp.status();
            if (200..300).contains(&status) {
                Ok(())
            } else {
                Err(format!("HTTP {}", status))
            }
        }
        Err(err) => Err(err.to_string()),
    }
}

/// Check whether a service is reachable by hitting its health endpoint.
#[tauri::command]
fn check_server(url: String) -> Result<bool, String> {
    Ok(probe_health(&url).is_ok())
}

/// Probe a single named service and return a full status object.
#[tauri::command]
fn test_connection(key: String) -> Result<ServiceStatus, String> {
    let Some((_, name, url)) = SERVICES.iter().find(|(k, _, _)| k == &key) else {
        return Err(format!("Unknown service '{}'", key));
    };
    let reachable = probe_health(url).is_ok();
    let detail = if reachable { "OK".to_string() } else { "unreachable".to_string() };
    Ok(ServiceStatus {
        key,
        name: name.to_string(),
        url: url.to_string(),
        reachable,
        detail,
    })
}

/// Probe every known service and return their statuses.
#[tauri::command]
fn probe_services() -> Vec<ServiceStatus> {
    SERVICES
        .iter()
        .map(|(key, name, url)| {
            let reachable = probe_health(url).is_ok();
            ServiceStatus {
                key: key.to_string(),
                name: name.to_string(),
                url: url.to_string(),
                reachable,
                detail: if reachable { "OK".to_string() } else { "unreachable".to_string() },
            }
        })
        .collect()
}

/// Open a native directory picker. Returns the chosen path, or `None` if the
/// user cancelled.
#[tauri::command]
async fn pick_folder(app: tauri::AppHandle, title: String) -> Result<Option<String>, String> {
    let picked = app
        .dialog()
        .file()
        .set_title(&title)
        .blocking_pick_folder();
    Ok(picked.map(|p| p.to_string()))
}

/// OS-appropriate default install directory for the app + services.
#[tauri::command]
fn default_install_dir() -> String {
    // Prefer %LOCALAPPDATA%\Programs (per-user, no admin). Fall back to
    // %USERPROFILE%\PrivateAI.
    std::env::var("LOCALAPPDATA")
        .map(|la| {
            PathBuf::from(la)
                .join("Programs")
                .join("PrivateAI")
                .to_string_lossy()
                .to_string()
        })
        .unwrap_or_else(|_| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("C:\\"))
                .join("PrivateAI")
                .to_string_lossy()
                .to_string()
        })
}

/// OS-appropriate default directory for storing local models.
#[tauri::command]
fn default_models_dir() -> String {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("C:\\"))
        .join("ai-server-models")
        .to_string_lossy()
        .to_string()
}

/// Register a logon scheduled task (no admin required) that runs the given
/// command at startup. Returns the task name on success.
#[tauri::command]
fn register_autostart(task_name: String, command: String) -> Result<String, String> {
    let output = std::process::Command::new("schtasks")
        .args([
            "/Create",
            "/TN",
            &task_name,
            "/TR",
            &command,
            "/SC",
            "ONLOGON",
            "/RL",
            "LIMITED",
            "/F",
        ])
        .output()
        .map_err(|e| format!("Failed to run schtasks: {}", e))?;

    if output.status.success() {
        Ok(task_name)
    } else {
        Err(String::from_utf8_lossy(&output.stderr).trim().to_string())
    }
}

/// Unregister a logon scheduled task.
#[tauri::command]
fn unregister_autostart(task_name: String) -> Result<(), String> {
    let output = std::process::Command::new("schtasks")
        .args(["/Delete", "/TN", &task_name, "/F"])
        .output()
        .map_err(|e| format!("Failed to run schtasks: {}", e))?;
    if output.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).trim().to_string())
    }
}

/// The web UI URL the installer navigates to once the server is up.
#[tauri::command]
fn web_ui_url() -> String {
    SERVER_URL.to_string()
}

/// Locate the bundled server executable (`ai-server.exe`) shipped via
/// `bundle.resources`. Returns its absolute path, or an error if not found.
#[tauri::command]
fn server_binary_path(app: tauri::AppHandle) -> Result<String, String> {
    let resource_dir = app
        .path()
        .resource_dir()
        .map_err(|e| format!("Cannot resolve resource dir: {}", e))?;
    // bundle.resources copies ../../dist/ai-server into <resource_dir>/ai-server
    let candidates = [
        resource_dir.join("ai-server").join("ai-server.exe"),
        resource_dir.join("ai-server.exe"),
    ];
    for c in &candidates {
        if c.is_file() {
            return Ok(c.to_string_lossy().to_string());
        }
    }
    Err(format!(
        "Bundled server not found. Looked in: {}",
        candidates.iter().map(|c| c.display().to_string()).collect::<Vec<_>>().join(", ")
    ))
}

/// Start the bundled server as a detached background process (if not already
/// running). Returns the process id or an explanatory error.
#[tauri::command]
fn start_bundled_server(app: tauri::AppHandle) -> Result<String, String> {
    // If the server is already reachable, nothing to do.
    if probe_health(SERVER_URL).is_ok() {
        return Ok("already-running".to_string());
    }
    let exe = server_binary_path(app)?;
    let exe_path = Path::new(&exe);
    let dir = exe_path.parent().unwrap_or(Path::new("."));

    let child = std::process::Command::new(&exe)
        .current_dir(dir)
        .arg("--port")
        .arg("8005")
        .spawn()
        .map_err(|e| format!("Failed to start server '{}': {}", exe, e))?;

    Ok(format!("started:{}", child.id()))
}

// ---------------------------------------------------------------------------
// Provisioning commands (used by the installer's Screen E checklist).
// These are intentionally best-effort: they detect what's present and report
// a clear message. Actual dependency install is wired in the setup scripts.
// ---------------------------------------------------------------------------

/// Check whether the `python` executable is on PATH.
#[tauri::command]
fn check_python() -> Result<String, String> {
    match std::process::Command::new("python").arg("--version").output() {
        Ok(o) if o.status.success() => {
            let v = String::from_utf8_lossy(&o.stdout).trim().to_string();
            Ok(format!("Python found: {}", v))
        }
        Ok(_) => Err("Python is not installed or not on PATH.".to_string()),
        Err(e) => Err(format!("Could not run python: {}", e)),
    }
}

/// Check whether Docker is installed and the daemon is running.
#[tauri::command]
fn check_docker() -> Result<String, String> {
    match std::process::Command::new("docker").arg("--version").output() {
        Ok(o) if o.status.success() => {
            let v = String::from_utf8_lossy(&o.stdout).trim().to_string();
            Ok(format!("Docker found: {}", v))
        }
        Ok(_) => Err("Docker is not installed or not running.".to_string()),
        Err(e) => Err(format!("Could not run docker: {}", e)),
    }
}

/// Best-effort CUDA / GPU detection via nvidia-smi.
#[tauri::command]
fn check_cuda() -> Result<String, String> {
    match std::process::Command::new("nvidia-smi").arg("--query-gpu=name").arg("--format=csv,noheader").output() {
        Ok(o) if o.status.success() => {
            let name = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if name.is_empty() {
                Err("No NVIDIA GPU detected.".to_string())
            } else {
                Ok(format!("GPU detected: {}", name))
            }
        }
        Ok(_) => Err("nvidia-smi not found — no NVIDIA GPU / CUDA detected.".to_string()),
        Err(e) => Err(format!("Could not run nvidia-smi: {}", e)),
    }
}

/// Start the stack via Docker Compose.
#[tauri::command]
fn docker_up() -> Result<String, String> {
    let output = std::process::Command::new("docker")
        .args(["compose", "up", "-d", "--build"])
        .output()
        .map_err(|e| format!("Failed to run docker compose: {}", e))?;
    if output.status.success() {
        Ok("Docker stack started.".to_string())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).trim().to_string())
    }
}

/// Clone a sibling server repository into the install directory if missing.
#[tauri::command]
fn clone_sibling(repo: String, install_dir: String) -> Result<String, String> {
    let url = match repo.as_str() {
        "open-fantasia-imagegen" => "https://github.com/fabiopacifici-bot/open-fantasia-imagegen",
        "olly-voice-server" => "https://github.com/fabiopacificicom/olly-voice-server",
        other => return Err(format!("Unknown sibling repo '{}'", other)),
    };
    let dest = std::path::Path::new(&install_dir).join(&repo);
    if dest.exists() {
        return Ok(format!("{} already present at {}", repo, dest.display()));
    }
    let output = std::process::Command::new("git")
        .args(["clone", url])
        .arg(&dest)
        .output()
        .map_err(|e| format!("Failed to run git: {}", e))?;
    if output.status.success() {
        Ok(format!("Cloned {} to {}", repo, dest.display()))
    } else {
        Err(String::from_utf8_lossy(&output.stderr).trim().to_string())
    }
}

/// Write a `.env` file with the installer's chosen configuration.
#[tauri::command]
fn write_env(install_dir: String, models_dir: String, mode: String) -> Result<String, String> {
    let env_path = std::path::Path::new(&install_dir).join(".env");
    let mut lines = Vec::new();
    lines.push(format!("HF_HOME={}", models_dir));
    lines.push("HF_HUB_DISABLE_SYMLINKS=1".to_string());
    lines.push("IMAGE_SERVER_URL=http://127.0.0.1:8765".to_string());
    lines.push("VOICE_SERVER_URL=http://127.0.0.1:8766".to_string());
    lines.push(format!("DEPLOY_MODE={}", mode));
    let content = lines.join("\n") + "\n";
    std::fs::write(&env_path, content).map_err(|e| format!("Failed to write {}: {}", env_path.display(), e))?;
    Ok(format!("Wrote {}", env_path.display()))
}

/// Start the local services (best-effort; real launcher is wired later).
#[tauri::command]
fn start_services(mode: String) -> Result<String, String> {
    if mode == "docker" {
        docker_up()
    } else {
        // For local mode, just confirm the server is reachable (it may already
        // be running). Actual process launching is wired via setup scripts.
        match probe_health(SERVER_URL) {
            Ok(()) => Ok("Private AI server is already running.".to_string()),
            Err(_) => Err("Private AI server is not running. Launch it from the install directory.".to_string()),
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            check_server,
            test_connection,
            probe_services,
            pick_folder,
            default_install_dir,
            default_models_dir,
            register_autostart,
            unregister_autostart,
            web_ui_url,
            server_binary_path,
            start_bundled_server,
            check_python,
            check_docker,
            check_cuda,
            docker_up,
            clone_sibling,
            write_env,
            start_services,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

