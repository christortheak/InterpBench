# Demo Workspaces

A Demo Workspace is a worked example: a finished study you can read before
downloading a model, and a draft copy of it that you can run yourself.

SteerLab never opens one in place. It makes a copy in a folder you choose,
checks the copy, and opens that:

- in the app, choose **Research Setup…** from the Workspace menu, then
  **Open Demo Workspace…**;
- from a command line, run `steerlab-cli workspace init <folder> --demo <backend>`
  on a Mac, or `steerlab workspace init <folder> --demo <backend>` with the
  cross-platform client.

There is one Demo Workspace per place a study can run, because a workspace is
tied to one engine and its vectors and results come from that engine:

| Backend | Where its studies run |
|---|---|
| `mlx` | This Mac, quick start: the engine built into the app |
| `mps` | This Mac, full capabilities: the Python engine on this Mac |
| `cuda` | Another machine: the Python engine on a workstation or a cluster |

A release may carry any of the three, or none. Asking for one it does not
carry gives a plain message, and nothing else changes.

## For people who build a Demo Workspace

Each one lives here as `<backend>/`: a workspace tree with a `README.md` and
a `demo.json` at its top.

`demo.json` names what the demo needs and holds:

```json
{
  "schemaVersion": 1,
  "backend": "mlx",
  "title": "A short name",
  "summary": "One line: what this demo shows.",
  "model": { "id": "owner/model", "approximateDownloadGB": 2.5 },
  "studies": [
    { "name": "the-study", "summary": "One line about this study." }
  ]
}
```

`backend` must match the folder's name, and every study named must be under
`experiments/`. Other keys are allowed and are passed through unchanged.

The tree must not contain:

- any file or folder whose name starts with a dot (`.git`, `.gitignore`,
  `.steerlab`). Opening a copy writes `.gitignore` and the workspace's compute
  setting itself;
- `AGENTS.md` or `WORKSPACE.md`. Opening a copy writes both fresh;
- symbolic links, or anything over the size limits: 8 MB for one Demo
  Workspace and 4 MB for one file.

A copy also receives any standard new-workspace file the tree does not carry,
so a demo needs to hold only its own content and the files its studies pin.

Check a tree before committing it:

```sh
python3 scripts/ci/check-demo-workspaces.py
```

The two test suites then open every Demo Workspace found here, compare the copy
byte for byte, and verify each study in the copy.
