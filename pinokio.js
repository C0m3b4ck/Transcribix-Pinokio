const path = require('path')
module.exports = {
  version: "1.0",
  title: "Transcribix",
  description: "Offline speech-to-text with 11 local AI models",
  icon: "icon.png",
  menu: async (kernel, info) => {
    let local = info.local("state") || {}
    let group = local.group

    let running = {
      install: info.running("install.js"),
      start: info.running("start.js"),
      update: info.running("update.js"),
      reset: info.running("reset.js"),
    }

    if (running.install) {
      return [{
        default: true,
        icon: "fa-solid fa-plug",
        text: "Installing " + (group || "") + "...",
        href: "install.js",
      }]
    } else if (running.start) {
      let url = info.local("start.js")
      if (url && url.url) {
        return [{
          default: true,
          icon: "fa-solid fa-rocket",
          text: "Open Web UI",
          href: url.url,
        }, {
          icon: 'fa-solid fa-terminal',
          text: "Terminal",
          href: "start.js",
        }]
      } else {
        return [{
          default: true,
          icon: 'fa-solid fa-terminal',
          text: "Terminal",
          href: "start.js",
        }]
      }
    } else if (running.update) {
      return [{
        default: true,
        icon: 'fa-solid fa-terminal',
        text: "Updating...",
        href: "update.js",
      }]
    } else if (running.reset) {
      return [{
        default: true,
        icon: 'fa-solid fa-terminal',
        text: "Resetting...",
        href: "reset.js",
      }]
    } else if (!group) {
      return [{
        default: true,
        icon: "fa-solid fa-microchip",
        text: "<div><strong>Whisper Variants</strong><div>faster-whisper, WhisperX, stable-ts, Distil-Whisper, Whisper, whisper.cpp</div></div>",
        href: "install.js?group=whisper",
      }, {
        icon: "fa-solid fa-microchip",
        text: "<div><strong>NVIDIA Models</strong><div>Parakeet TDT, Canary Qwen (best accuracy)</div></div>",
        href: "install.js?group=nvidia",
      }, {
        icon: "fa-solid fa-microchip",
        text: "<div><strong>Lightweight Models</strong><div>Moonshine, SenseVoice, Vosk (minimal resources)</div></div>",
        href: "install.js?group=lightweight",
      }, {
        icon: "fa-solid fa-microchip",
        text: "<div><strong>All Models</strong><div>Install all 11 models (requires ~20GB disk)</div></div>",
        href: "install.js?group=all",
      }]
    } else {
      let installed = info.exists("venvs/" + group)
      let groupNames = {
        whisper: "Whisper Variants",
        nvidia: "NVIDIA Models",
        lightweight: "Lightweight Models",
        all: "All Models",
      }
      let displayName = groupNames[group] || group

      if (installed) {
        return [{
          default: true,
          icon: "fa-solid fa-power-off",
          text: "Start " + displayName,
          href: "start.js?group=" + group,
        }, {
          icon: "fa-solid fa-plug",
          text: "Update " + displayName,
          href: "update.js?group=" + group,
        }, {
          icon: "fa-solid fa-plug",
          text: "Reinstall " + displayName,
          href: "install.js?group=" + group,
        }, {
          icon: "fa-regular fa-circle-xmark",
          text: "<div><strong>Reset " + displayName + "</strong><div>Revert to pre-install state</div></div>",
          href: "reset.js?group=" + group,
          confirm: "Are you sure you wish to reset " + displayName + "?"
        }, {
          icon: "fa-solid fa-arrow-left",
          text: "Change Model Group",
          href: "resetgroup.js",
        }]
      } else {
        return [{
          default: true,
          icon: "fa-solid fa-plug",
          text: "Install " + displayName,
          href: "install.js?group=" + group,
        }, {
          icon: "fa-solid fa-arrow-left",
          text: "Change Model Group",
          href: "resetgroup.js",
        }]
      }
    }
  }
}
