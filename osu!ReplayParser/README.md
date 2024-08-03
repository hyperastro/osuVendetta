# osu! replay parser
Converts .osr files to their LLM tokens in .txt format

---

# Requirements
[.Net 8 SDK](https://www.dotnet.microsoft.com/en-us/download/dotnet/8.0)

(If you use pre-compiled binaries the [.Net 8 Runtime](https://www.dotnet.microsoft.com/en-us/download/dotnet/8.0)
 will be sufficient)

---

# Compiling

1. Open your powershell/cmd
2. Navigate into the the replay parser directory
3. Use the command: ``dotnet build --configuration Release``
4. Inside ``./bin/release/`` you will find the ``osu!ReplayParser.exe`` binary 

---

# Usage

<> = required
[] = optional

- .\osu!ReplayParser.exe <Directories/File> [-out Directory]

You can supply a unlimited amount of directories and/or files.
The ``-out`` parameter is used for the output folder and is **optional**
if ``-out`` is **not** supplied it will default to ``.\out\`` 

e.g.:
- .\osu!ReplayParser.exe "Src Dir" -out "Out Dir"
- .\osu!ReplayParser.exe "Src Dir"
- .\osu!ReplayParser.exe "Src File" -out "Out Dir"
- .\osu!ReplayParser.exe "Src Dir1" "Src Dir2" -out "Out Dir"
