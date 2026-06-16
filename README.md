# OpenSpineToolbox

A shared collection of student miniprojects for spine & pelvis imaging analysis.
Each miniproject lives in its own folder under [`projects/`](projects/) and is
contributed through a pull request.

This README is your **step-by-step guide to adding your project.** No prior Git
experience is assumed — follow it top to bottom.

---

## How contributing works (the big picture)

You do **not** push directly to this repository. Instead you:

1. **Fork** it → this gives you your own copy on GitHub.
2. **Clone** your fork → a copy on your laptop you can edit.
3. **Branch** → an isolated workspace for your miniproject.
4. **Add your code** in a new folder, then **commit** and **push** it to your fork.
5. **Open a pull request (PR)** → asks us to merge your folder into the main repo.

We review the PR and merge it. Your code then appears in the main toolbox for
everyone. This is the standard open-source workflow — learning it is part of the
project.

---

## Step 0 — Create a GitHub account & install Git

1. **Create a GitHub account:** go to <https://github.com> → **Sign up**.
2. **Install Git:**
   - **Windows:** download from <https://git-scm.com/download/win> (accept the
     defaults; this also installs "Git Bash", a terminal you can use for every
     command below).
   - **macOS:** open Terminal and run `git --version` — if Git isn't installed it
     will offer to install it. Or install from <https://git-scm.com/download/mac>.
   - **Linux:** `sudo apt install git` (Debian/Ubuntu) or your distro's equivalent.
3. **Tell Git who you are** (run once, in your terminal):
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "you@example.com"
   ```

> **Authentication note:** GitHub no longer accepts your account *password* on the
> command line. The first time you `push`, you'll need either the **GitHub CLI**
> (`gh auth login`, easiest — <https://cli.github.com>) or a **Personal Access
> Token** used in place of your password (GitHub → Settings → Developer settings →
> Personal access tokens → *Tokens (classic)* → scope `repo`). See
> [Troubleshooting](#troubleshooting) if a push asks for a password.

---

## Step 1 — Fork the main repository

1. Go to the main repo: <https://github.com/Gregory-Schwing-MD-PhD/OpenSpineToolbox>
2. Click **Fork** (top-right) → **Create fork**.
3. You now have your own copy at
   `https://github.com/<your-username>/OpenSpineToolbox`.

---

## Step 2 — Clone *your fork* to your computer

Copy the URL of **your** fork (green **Code** button → HTTPS), then:

```bash
git clone https://github.com/<your-username>/OpenSpineToolbox.git
cd OpenSpineToolbox
```

Replace `<your-username>` with your actual GitHub username.

---

## Step 3 — Create a branch for your miniproject

Never work on `main`. Make a branch named after your project:

```bash
git checkout -b miniproject-<short-name>
```

Example: `git checkout -b miniproject-cobb-angle`.

---

## Step 4 — Add your code to your project folder

**Your project folder already exists under [`projects/`](projects/)** — find the
one matching your miniproject and drop your code into it. Each folder has a
`README.md` describing the project and how the dataset helps; fill in the blanks
at the bottom. (If your project isn't listed, create a new folder — lowercase,
hyphens, no spaces.)

```
projects/
└── cobb-angle/
    ├── README.md        <- what it does + how to run it (required)
    ├── main.py          <- your code
    └── ...              <- any other files
```

**Every project folder must include its own `README.md`** stating:
- what the project does,
- how to run it (dependencies + the command),
- your name / team.

> Please commit **code and small files only** — no large datasets, patient data,
> or DICOM/NIfTI volumes. If you need data, link to it in your README instead.

---

## Step 5 — Commit and push to your fork

Stage **only your project folder** (don't use `git add -A`), commit, and push:

```bash
git add projects/cobb-angle
git commit -m "Add Cobb angle miniproject"
git push -u origin miniproject-<short-name>
```

(The first `push` is where GitHub asks you to authenticate — see Step 0's note.)

---

## Step 6 — Open a pull request onto the main repo

1. Go to **your fork** on GitHub. A yellow banner appears:
   **"Compare & pull request"** — click it.
   (If you don't see it: **Pull requests** tab → **New pull request**.)
2. Make sure the PR is set up as:
   - **base repository:** `Gregory-Schwing-MD-PhD/OpenSpineToolbox`, **base:** `main`
   - **head repository:** `<your-username>/OpenSpineToolbox`, **compare:**
     `miniproject-<short-name>`
3. Give it a clear title (e.g. *"Add Cobb angle miniproject"*) and a short
   description of what your project does.
4. Click **Create pull request**.

That's it — we'll review it, maybe ask for small changes, and merge it. 🎉

---

## Updating your PR after feedback

If we request changes, you don't open a new PR — just push more commits to the
same branch and the PR updates automatically:

```bash
# make your edits, then:
git add projects/cobb-angle
git commit -m "Address review feedback"
git push
```

---

## Troubleshooting

- **`push` asks for a username/password, then rejects my password.** GitHub
  removed password auth. Use `gh auth login` (GitHub CLI) once, or paste a
  **Personal Access Token** (not your password) when prompted. See Step 0.
- **`Permission denied` / `403` on push.** You're trying to push to the *main*
  repo instead of *your fork*. Check `git remote -v` — `origin` must point at
  `https://github.com/<your-username>/OpenSpineToolbox.git`.
- **I accidentally committed to `main`.** Make your branch from where you are
  (`git checkout -b miniproject-<short-name>`) and push that; the PR will still
  work.
- **Stuck?** Open an issue on the main repo or ask in class.

---

## Repository layout

```
OpenSpineToolbox/
├── LICENSE
├── README.md            <- you are here
└── projects/
    └── <your-project>/  <- one folder per miniproject
```
