import requests

# API endpoint
url = 'http://127.0.0.1:5000/predict'

# Sample data
data = {
    'title': 'Specter of Trump Loosens Tongues, if Not Purse Strings, in Silicon Valley - The New York Times',
    'author': 'David Streitfeld',
    'text': """
    If at first you don’t succeed, try a different sport. 
    Tim Tebow, who was a Heisman quarterback at the University of Florida but was unable to hold an N. F. L. job, is pursuing a career in Major League Baseball. 
    He will hold a workout for M. L. B. teams this month, his agents told ESPN and other news outlets. 
    “This may sound like a publicity stunt, but nothing could be further from the truth,” said Brodie Van Wagenen,
    of CAA Baseball, part of the sports agency CAA Sports, in the statement. 
    “I have seen Tim’s workouts, and people inside and outside the industry  —   scouts, executives, players and fans  —   will be impressed by his talent. 
    ” It’s been over a decade since Tebow, 28, has played baseball full time, which means a comeback would be no easy task. 
    But the former major league catcher Chad Moeller, who said in the statement that he had been training Tebow in Arizona, said he was “beyond impressed with Tim’s athleticism and swing. ” “I see bat speed and power and real baseball talent,” Moeller said. 
    “I truly believe Tim has the skill set and potential to achieve his goal of playing in the major leagues and based on what I have seen over the past two months, it could happen relatively quickly. ” Or, take it from Gary Sheffield, the former   outfielder. 
    News of Tebow’s attempted comeback in baseball was greeted with skepticism on Twitter. 
    As a junior at Nease High in Ponte Vedra, Fla. Tebow drew the attention of major league scouts, batting . 
    494 with four home runs as a left fielder. 
    But he ditched the bat and glove in favor of pigskin, leading Florida to two national championships, in 2007 and 2009. 
    Two former scouts for the Los Angeles Angels told WEEI, a Boston radio station, that Tebow had been under consideration as a high school junior. 
    “’x80’x9cWe wanted to draft him, ’x80’x9cbut he never sent back his information card,” said one of the scouts, Tom Kotchman, referring to a questionnaire the team had sent him. “He had a strong arm and had a lot of power,” said the other scout, Stephen Hargett. “If he would have been there his senior year he definitely would have had a good chance to be drafted. ” “It was just easy for him,” Hargett added. “You thought, If this guy dedicated everything to baseball like he did to football how good could he be?” Tebow’s high school baseball coach, Greg Mullins, told The Sporting News in 2013 that he believed Tebow could have made the major leagues. “He was the leader of the team with his passion, his fire and his energy,” Mullins said. “He loved to play baseball, too. He just had a bigger fire for football. ” Tebow wouldn’t be the first athlete to switch from the N. F. L. to M. L. B. Bo Jackson had one   season as a Kansas City Royal, and Deion Sanders played several years for the Atlanta Braves with mixed success. Though Michael Jordan tried to cross over to baseball from basketball as a    in 1994, he did not fare as well playing one year for a Chicago White Sox minor league team. As a football player, Tebow was unable to match his college success in the pros. The Denver Broncos drafted him in the first round of the 2010 N. F. L. Draft, and he quickly developed a reputation for clutch performances, including a memorable   pass against the Pittsburgh Steelers in the 2011 Wild Card round. But his stats and his passing form weren’t pretty, and he spent just two years in Denver before moving to the Jets in 2012, where he spent his last season on an N. F. L. roster. He was cut during preseason from the New England Patriots in 2013 and from the Philadelphia Eagles in 2015.
    """,
    'algorithm': 'xgboost'  # Choose 'xgboost', 'random_forest', or 'pycaret'
}

# Send POST request
response = requests.post(url, json=data)

# Print response
result = response.json()
if result.get('prediction') is not None:
    if result['prediction']:
        print("The news is not fake.")
    else:
        print("The news is fake.")
elif result.get('error') is not None:
    print("Error:", result['error'])
else:
    print("Unexpected response received.")
