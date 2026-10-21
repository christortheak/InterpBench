# Set up SteerLab with your coding assistant

**For the researcher.** Give your coding assistant this page's address and
say what you want, for example:

> Set up SteerLab for me by following
> https://github.com/christortheak/InterpBench/blob/main/docs/AGENT-SETUP.md
> and then help me design my first study.

It shows you a plan and asks before it downloads or installs anything. On a
Mac it installs the SteerLab app, creates a workspace for your study, and
offers to open the app on it. On Linux it sets up the app-free client.

---

**For the coding assistant.** Follow these steps in order. They install
SteerLab for the researcher who gave you this page, and then hand you to the
workspace's own AGENTS.md. This page needs SteerLab 0.9.14 or later; step 2
says what to do with an older copy.

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
- **Each code block below stands alone.** It sets every variable it uses,
  because your tool may run each command in a fresh shell.
- **If you cannot run commands or reach the network,** give the researcher
  the manual steps at the end of this page instead.

## 1. Check the computer

```sh
uname -s                      # Darwin is a Mac, Linux is Linux
sysctl -n hw.optional.arm64   # on a Mac: 1 means Apple Silicon, even under Rosetta
sw_vers -productVersion       # on a Mac: needs 26.4 or later
```

The SteerLab app needs an Apple Silicon Mac with macOS 26.4 or later. On
x86_64 Linux, set up the app-free client instead (go to "On Linux" below).
Windows is not supported: tell the researcher so, and stop.

## 2. Look for an existing copy

```sh
for APP in "$HOME/SteerLab/SteerLab.app" /Applications/SteerLab.app; do
  [ -d "$APP" ] && "$APP/Contents/Helpers/steerlab-cli" --version | head -1 && echo "  at $APP"
done
```

If SteerLab is already installed, do not download or replace it. Updating is
the researcher's click on Install Update in the app. Use that copy, by the
path printed, wherever the steps below say `$HOME/SteerLab/SteerLab.app`, and
go to step 4.

If its version is older than 0.9.14, ask the researcher to update first:
open SteerLab and choose **Check for Updates…** in the SteerLab menu, then
**Install Update**. If they would rather not, you can still continue, with
two differences: skip `install link` in step 4, and in step 6 ask them to
choose **Open Workspace…** in the app's Workspace menu instead of using
`openInApp`.

## 3. Show the plan, then download and verify

Ask GitHub which release is current:

```sh
D="${TMPDIR:-/tmp}/steerlab-setup"; mkdir -p "$D"
curl -fsSL https://api.github.com/repos/christortheak/InterpBench/releases/latest -o "$D/release.json"
grep -o '"tag_name": *"[^"]*"' "$D/release.json"
grep -o '"browser_download_url": *"[^"]*/SteerLab-[^"]*\.zip"' "$D/release.json"
```

Tell the researcher the version, that the app download is about 65 MB (the
release page lists the exact size), and that it will be installed as
`~/SteerLab/SteerLab.app`, which needs no administrator password. Nothing
else is installed at this step. When they agree, download the app and its
checksum, naming the files yourself (the release's file names contain a `+`,
which the download address spells `%2B`):

```sh
D="${TMPDIR:-/tmp}/steerlab-setup"
APP_URL='<the SteerLab-….zip address printed above>'
case "$APP_URL" in
  https://github.com/christortheak/InterpBench/releases/download/*)
    curl -fL -o "$D/SteerLab.zip" "$APP_URL" &&
      curl -fL -o "$D/SteerLab.zip.sha256" "$APP_URL.sha256" &&
      expected=$(awk '{print $1}' "$D/SteerLab.zip.sha256") &&
      actual=$(shasum -a 256 "$D/SteerLab.zip" | awk '{print $1}') &&
      [ -n "$expected" ] && [ "$expected" = "$actual" ] && echo "checksum OK" ;;
  *) echo "not this project's release address: stop here" ;;
esac
```

Unpack it and check who signed it before it goes anywhere:

```sh
D="${TMPDIR:-/tmp}/steerlab-setup"
rm -rf "$D/unpacked" && ditto -x -k "$D/SteerLab.zip" "$D/unpacked"
codesign --verify --deep --strict \
  -R='anchor apple generic and identifier "org.steerlab.SteerLab" and certificate leaf[subject.OU] = "S7FM9Y7393"' \
  "$D/unpacked/SteerLab.app" && echo "signer OK"
spctl --assess --type execute -vv "$D/unpacked/SteerLab.app"
```

Continue only if `checksum OK` and `signer OK` were printed and `spctl` says
`accepted` with `source=Notarized Developer ID`. The signer check requires
Apple's certificate chain, the app's identifier, and this project's
Developer ID team, `S7FM9Y7393`. Then install it:

```sh
D="${TMPDIR:-/tmp}/steerlab-setup"
mkdir -p "$HOME/SteerLab"
mv "$D/unpacked/SteerLab.app" "$HOME/SteerLab/SteerLab.app"
```

Keep it there: the app updates itself in place, and it cannot from a
temporary or read-only location.

## 4. Check the command line

```sh
"$HOME/SteerLab/SteerLab.app/Contents/Helpers/steerlab-cli" --version
```

It must report the release's version and `7/7 resource families resolved`.
Run this command line by its full path for the rest of the setup. Never use a
`steerlab-cli` from a build folder or a source checkout, and do not read or
change anything else inside SteerLab.app.

Optionally, offer to put it on the researcher's PATH, so later sessions find
it by name:

```sh
"$HOME/SteerLab/SteerLab.app/Contents/Helpers/steerlab-cli" install link --json
```

It links `~/.local/bin/steerlab-cli` to this app and says whether
`~/.local/bin` is on PATH. Add the line it gives to a shell profile only if
the researcher agrees. If it refuses because a link to another copy of
SteerLab is there, ask before running it again with `--replace`; it never
replaces anything that is not SteerLab's. Later sessions can also find the
command line through the workspace, so this step is a convenience.

## 5. Create the study's workspace

Ask the researcher for a short name for the study, such as `first-study`.
Then create SteerLab's home folders (this never overwrites anything), the
workspace, and your handoff:

```sh
CLI="$HOME/SteerLab/SteerLab.app/Contents/Helpers/steerlab-cli"
W="$HOME/SteerLab/Workspaces/<study-name>"
"$CLI" init --json
"$CLI" workspace init "$W" --json
"$CLI" --workspace "$W" workspace handoff --json
```

Studies are designed with a small helper that installs separately. Check it:

```sh
"$HOME/SteerLab/SteerLab.app/Contents/Helpers/steerlab-cli" setup inspect --json
```

If `result.authoringReady` is false, run `setup plan --json` with the same
command line and show the researcher the plan, including
`result.approximateDownloadMB`. Only after they agree, install it with
`setup apply --expect <result.planSHA256> --yes --json`.

## 6. Hand over to the workspace

Read `AGENTS.md` in the new workspace and follow it from here. Begin with the
study interview named in the handoff's `instructions`, at the researcher's
level and before any model is chosen or downloaded. Workspace data stays in
that folder.

Offer to open the app on the workspace. The handoff's `openInApp` is the
exact command; it has this form:

```sh
open -a "$HOME/SteerLab/SteerLab.app" "$HOME/SteerLab/Workspaces/<study-name>"
```

The app opens on that workspace. Research Setup is in its Workspace menu. The
larger engine for full capabilities on this Mac (about 2 GB) is set up from
the app, which shows its own plan first.

## On Linux

The app-free client authors studies and submits them to a runner; it does
not run models itself. With the researcher's agreement, download the client
and its checksum from the
[latest release](https://github.com/christortheak/InterpBench/releases/latest)
(`steerlab-client-<version>+<revision>.tar.gz` and its `.sha256`, named
yourself as in step 3), check that `sha256sum` of the archive matches the
first field of the `.sha256` file, extract it, and follow the `AGENTS.md`
inside it.

## If you cannot run these steps

Give the researcher these steps instead:

1. Open the [latest release](https://github.com/christortheak/InterpBench/releases/latest)
   and download `SteerLab-<version>+<revision>.zip`.
2. Unzip it, make a folder named `SteerLab` in your home folder, and move
   `SteerLab.app` into it.
3. Open SteerLab and follow Research Setup to create a workspace.
4. Choose **Copy Instructions for Your Coding Assistant**, open the workspace
   folder in your coding assistant, and paste them.
