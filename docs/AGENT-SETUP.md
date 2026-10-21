# Set up SteerLab with your coding assistant

**For the researcher.** Give your coding assistant this page's address and
say what you want, for example:

> Set up SteerLab for me by following
> https://github.com/christortheak/InterpBench/blob/main/docs/AGENT-SETUP.md
> and then help me design my first study.

It shows you a plan and asks before it downloads or installs anything. On a
Mac it installs the SteerLab app, creates a workspace for your study, and
offers to open the app on it. Elsewhere it sets up the app-free client.

---

**For the coding assistant.** Follow these steps in order. They install
SteerLab for the researcher who gave you this page, and then hand you to the
workspace's own AGENTS.md.

## Rules for this setup

- **Ask before each download or install,** and say what it is and how large.
  The researcher's request is permission to follow this page, not to skip
  its questions.
- **Use only this project's GitHub Releases**
  (`github.com/christortheak/InterpBench`) and verify every file as shown.
  If a check fails, stop and show the researcher the output.
- **Never use `sudo`,** never change Gatekeeper or other security settings,
  and never remove a file's quarantine attribute. Nothing here needs them.
- **Never put a password or token** in a file or a command.
- **If you cannot run commands or reach the network,** give the researcher
  the manual steps at the end of this page instead.

## 1. Check the computer

```sh
uname -s -m                  # Darwin arm64 means an Apple Silicon Mac
sw_vers -productVersion      # on a Mac: 26.4 or later
```

The SteerLab app needs an Apple Silicon Mac with macOS 26.4 or later. On any
other computer, set up the app-free client instead: download
`steerlab-client-<version>+<revision>.tar.gz` from the
[latest release](https://github.com/christortheak/InterpBench/releases/latest),
extract it, and follow the `AGENTS.md` inside it. That client authors studies
and submits them to a runner; it does not run models itself.

## 2. Look for an existing copy

```sh
ls -d ~/SteerLab/SteerLab.app /Applications/SteerLab.app 2>/dev/null
```

If SteerLab is already installed, do not download or replace it. Updating is
the researcher's click on Install Update in the app. Use that copy's command
line, `<app>/Contents/Helpers/steerlab-cli`, and go to step 5.

## 3. Show the plan, then download and verify

Ask GitHub which release is current:

```sh
T=$(mktemp -d)
curl -fsSL https://api.github.com/repos/christortheak/InterpBench/releases/latest -o "$T/release.json"
grep -o '"tag_name": *"[^"]*"' "$T/release.json"
grep -o '"browser_download_url": *"[^"]*/SteerLab-[^"]*\.zip"' "$T/release.json"
```

Tell the researcher the version, that the app download is about 65 MB, and
that it will be installed as `~/SteerLab/SteerLab.app`, which needs no
administrator password. Nothing else is installed at this step. When they
agree, download the app and its checksum, naming the files yourself (the
release's file names contain a `+`, which the download address spells `%2B`):

```sh
APP_URL='<the SteerLab-….zip address printed above>'
curl -fL -o "$T/SteerLab.zip" "$APP_URL"
curl -fL -o "$T/SteerLab.zip.sha256" "$APP_URL.sha256"
expected=$(awk '{print $1}' "$T/SteerLab.zip.sha256")
actual=$(shasum -a 256 "$T/SteerLab.zip" | awk '{print $1}')
[ -n "$expected" ] && [ "$expected" = "$actual" ] && echo "checksum OK"
```

Unpack it and check who signed it before it goes anywhere:

```sh
ditto -x -k "$T/SteerLab.zip" "$T/unpacked"
codesign --verify --deep --strict "$T/unpacked/SteerLab.app" && echo "signature OK"
codesign -dv "$T/unpacked/SteerLab.app" 2>&1 | grep -E '^(Identifier|TeamIdentifier)='
spctl --assess --type execute -vv "$T/unpacked/SteerLab.app"
```

Continue only if all of these hold:

- `checksum OK` and `signature OK` were printed;
- `Identifier=org.steerlab.SteerLab` and `TeamIdentifier=S7FM9Y7393`;
- `spctl` says `accepted` and `source=Notarized Developer ID`.

Then install it:

```sh
mkdir -p ~/SteerLab
mv "$T/unpacked/SteerLab.app" ~/SteerLab/SteerLab.app
```

Keep it there: the app updates itself in place, and it cannot from a
temporary or read-only location.

## 4. Check the command line

```sh
CLI="$HOME/SteerLab/SteerLab.app/Contents/Helpers/steerlab-cli"
"$CLI" --version
```

It must report the release's version and `7/7 resource families resolved`.
Use this command line by its full path for the rest of the setup. Never use a
`steerlab-cli` from a build folder or a source checkout, and do not read or
change anything else inside SteerLab.app.

Optionally, offer to put it on the researcher's PATH, so later sessions find
it by name:

```sh
"$CLI" install link --json
```

It links `~/.local/bin/steerlab-cli` to this app and says whether
`~/.local/bin` is on PATH. Add the line it gives to a shell profile only if
the researcher agrees. Later sessions can also find the command line through
the workspace, so this step is a convenience.

## 5. Create the study's workspace

Ask the researcher for a short name for the study, such as `first-study`.
Then create SteerLab's home folders (this never overwrites anything) and the
workspace:

```sh
"$CLI" init --json
"$CLI" setup start ~/SteerLab/Workspaces/<study-name> --create --json
```

`setup start` returns `result.readiness` and `result.handoff`. If readiness
says the small study-design helper is not installed, run
`"$CLI" setup plan --json` and show the researcher the plan, including
`result.approximateDownloadMB`. Only after they agree, install it with
`"$CLI" setup apply --expect <result.planSHA256> --yes --json`.

## 6. Hand over to the workspace

Read `AGENTS.md` in the new workspace and follow it from here. Begin with the
study interview named in `result.handoff.instructions`, at the researcher's
level and before any model is chosen or downloaded. Workspace data stays in
that folder.

Offer to open the app on the workspace. The handoff's `openInApp` is the
exact command; it has this form:

```sh
open -a ~/SteerLab/SteerLab.app ~/SteerLab/Workspaces/<study-name>
```

The app opens on that workspace. Research Setup is in its Workspace menu. The
larger engine for full capabilities on this Mac (about 2 GB) is set up from
the app, which shows its own plan first.

## If you cannot run these steps

Give the researcher these steps instead:

1. Open the [latest release](https://github.com/christortheak/InterpBench/releases/latest)
   and download `SteerLab-<version>+<revision>.zip`.
2. Unzip it, make a folder named `SteerLab` in your home folder, and move
   `SteerLab.app` into it.
3. Open SteerLab and follow Research Setup to create a workspace.
4. Choose **Copy Instructions for Your Coding Assistant**, open the workspace
   folder in your coding assistant, and paste them.
