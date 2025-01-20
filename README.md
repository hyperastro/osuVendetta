# osu! Vendetta

Welcome to the osu! Vendetta!

Our goal is to train an AI model (currently Bi-LSTM model) to analyze osu! replays and catch cheaters with a high degree of accuracy.

How You Can Contribute: 

    Submit Replays: Help us improve our model by submitting both cheated and non-cheated replays on our Discord server.
    Join our Discord: https://discord.gg/BDU3W22HEW

    Contribute to the Codebase: If you're a developer, your contributions to our codebase would be greatly appreciated.
   
By participating, you'll be playing a crucial role in making osu! a fairer and more enjoyable game for everyone. Thank you for your support!

Code will be uploaded as soon as its confirmed to be stable if you want to contribute now DM me on discord


---

Note
- ``osuVendetta.\*`` refers to the folder/project structure.
    - Where ``*`` represents any folder starting with ``osuVendetta.``


## 1. Requirements

**osu! report web scraper script**

- Python 3.10
- Reddit API key
- Depdencies:
    - ``pip install praw``

**osuVendetta.\***

- .Net 9
- Preferably ``Visual Studio`` ``Community Edition`` (Enable ``Preview`` mode)

## 2. Setup

**osu! report web scraper script**

- TBD

**osuVendetta.\***

- TBD

## 3. Running

**osu! report web scraper script**

- Input your api key in the script and run it.
    - It should return the scoreID of the play followed by the type of cheat
    - you can change the number of messages scraped by changing the limmit value, it is set to 100 by default

**osuVendetta.\***

- Start ``osuVendetta.Cli.exe``
