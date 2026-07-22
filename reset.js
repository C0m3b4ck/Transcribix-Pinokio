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
        description: "The virtual environment has been removed."
      }
    },
  ]
}
