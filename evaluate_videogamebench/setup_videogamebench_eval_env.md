# Setting up the VideoGameBench evaluation environment

**VideoGameBench is not included in this repo.** You need the external VideoGameBench repository.

**DOS games only** — Game Boy / PyBoy games are excluded (no ROMs or emulator required).

## 1. VideoGameBench repository (external)

Clone and install VideoGameBench as a sibling of Game-AI-Agent (or elsewhere and add to `PYTHONPATH`):

```powershell
cd D:\ICML2026
git clone <videogamebench-repo-url>
cd videogamebench
pip install -e .
```

## 2. Dependencies

```powershell
pip install playwright
playwright install chromium
```

DOS games run in a browser (Playwright + JS-DOS). No ROMs needed.

## 3. Environment variables for LLM APIs

```powershell
$env:OPENAI_API_KEY = "your_api_key"
```

## 4. Run evaluation tests

From the Game-AI-Agent codebase root:

```powershell
cd D:\ICML2026\Game-AI-Agent
$env:PYTHONPATH = (Get-Location).Path + ";" + (Get-Location).Path + "\..\videogamebench"
python evaluate_videogamebench/test_videogamebench_dummy.py --game doom2 --max_steps 20
python evaluate_videogamebench/test_videogamebench_dummy.py --game doom2 --mode random_nl --headless
python evaluate_videogamebench/test_videogamebench_dummy.py --list-games
```

Available DOS games (run `--list-games`): doom2, doom, quake, civ, warcraft2, oregon_trail, x-com, incredible-machine, prince, need_for_speed, age_of_empires, comanche2.
