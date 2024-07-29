import praw
import re

#Put your reddit api auth below 
client_id = '' 
client_secret = ''
user_agent = ''


reddit = praw.Reddit(client_id=client_id,
                     client_secret=client_secret,
                     user_agent=user_agent)

subreddit = reddit.subreddit('osureport')

osuLink = re.compile(r'https://osu.ppy.sh/scores/osu/(\d+)')

exclude_terms = ['multi-accounting', 'multiaccounting', 'multi accounting','account sharing']

def OsuScoreScraper():
    scoreID_set = set()

    for i in subreddit.new(limit=100):
        if '[osu!std]' in i.title:
            post_text = i.selftext.lower()
            if not any(term.lower() in post_text for term in exclude_terms):
                matches = osuLink.findall(i.selftext)
                parts = i.title.split('|')
                additional_info = parts[1].strip() if len(parts) > 1 else ''
                for match in matches:
                    scoreID_set.add((match, additional_info))

    return scoreID_set

osu_score_info_set = OsuScoreScraper()
for score_id, info in osu_score_info_set:
    print(f"Score ID: {score_id}, Info: {info}")
