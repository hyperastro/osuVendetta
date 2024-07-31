# osu!-AI-Anti-Cheat-Project

Welcome to the osu!-AI-Anti-Cheat Project!

Our goal is to train an AI model (using BERT, known for its bi-directional tokenization) to analyze osu! replays and catch cheaters with a high degree of accuracy.

How You Can Contribute: 

    Submit Replays: Help us improve our model by submitting both cheated and non-cheated replays on our Discord server.
    Join our Discord: https://discord.gg/BDU3W22HEW

    Contribute to the Codebase: If you're a developer, your contributions to our codebase would be greatly appreciated.
   
By participating, you'll be playing a crucial role in making osu! a fairer and more enjoyable game for everyone. Thank you for your support!

Code will be uploaded as soon as its confirmed to be stable if you want to contribute now DM me on discord




**Using the Osu report Web Scraper script**

the script was written in python 3.10

you will need to install the following dependencies 

    pip install praw

you will also need to request a API key from reddit

Then input your api key in the script and run it. it should return the scoreID of the play followed by the type of cheat

you can change the number of messages scraped by changing the limmit value, it is set to 100 by default
