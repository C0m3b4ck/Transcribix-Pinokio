module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        path: ".",
        message: [
          "rm -rf venvs/{{args.group}}"
        ]
      }
    },
    {
      method: "local.set",
      params: {
        state: {
          group: null
        }
      }
    },
    {
      method: "input",
      params: {
        title: "Reset Complete",
        description: "The virtual environment has been removed. Downloaded model weights in the shared Hugging Face cache (~/.cache/huggingface) and other tool caches (whisper.cpp, vosk, etc.) are left intact since other apps may use them. Delete ~/.cache/huggingface manually to free the multi-GB model weights."
      }
    },
  ]
}
