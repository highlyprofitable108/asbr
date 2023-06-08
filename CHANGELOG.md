**Changelog - "High" Level Overview, May 23rd, 2023**

**API Connection & Data Retrieval**
- Our genius main character, the code, had a passionate rendezvous with the DataGolf API, dancing gracefully to extract round scoring statistics, historical outrights, and matchups, all while serenading the API with consentful rate limiting. The code moved smoother than smoke from some freshly lit Blue Dream. 

**CSV Creation & Formatting**
- CSV files were birthed like a puff of smoke, each elegantly timestamped. However, we did run into a rebellious streak of formatting issues. But no worries, with some wizard-like tricks, the code made the CSVs fall in line, just like that perfectly rolled joint. 

**Database Connection & Data Migration**
- Our trusty SQLite database, acting like a munchies-filled pantry, welcomed the raw data, outrights, and matchups, creating a repository as rich and satisfying as your favorite strain after a long day. 
- We then escorted our data from the CSV files into the database. This migration flowed smoother than smoke through a well-cleaned bong. 

**Data Cleaning**
- In the spirit of keeping things unique and spicy, like the varied strains of cannabis, the code swiftly banished any duplicates from our digital Eden.

**Project Status Update**
- And finally, the project status, a testament to our journey, got updated with a sprinkle of wit, reminding us all that we're not just coders, but storytellers, spinning tales through code under a haze of creativity.

**Changelog - "High" Level Overview, May 24th, 2023**

**API Connection & Data Retrieval**
- Another day of our code making magic with the DataGolf API, extracting all sorts of statistical nuggets, like finding the perfect bud in your bag of Pineapple Express. 

**CSV Creation & Formatting**
- CSV files unfurled, their timestamps shimmering like dew in the morning sun. The formatting issues were tamed, restoring harmony and making our CSV realm as peaceful as a meditative smoke sesh.

**Database Connection & Data Migration**
- The SQLite database bloomed, its petals of data unfurling in a display of digital beauty. Then, like a peaceful stream, our data flowed from CSV files into the database, a sight as serene as the gentle lapping of water against the edges of a well-used bong. 

**Data Cleaning**
- As we value diversity in our data like we appreciate variety in our strains, the code continued its mission to banish duplicates from the database. 

**Project Status Update**
- The project status, a poetry of our journey, got another witty update. Each word is a testament to our relentless pursuit of the extraordinary, spiced up with a dash of intoxicating humor.

**Changelog - "High" Level Overview, May 25th, 2023**

**Data Analysis**
- The code embarked on a psychedelic journey, diving deep into the data tables, extracting 'round_score' for each player in every event, and confronting missing rounds. It's like venturing into the vast ocean of data, guided by the ethereal glow of a Purple Haze strain.

**Model Preparation & Execution**
- Like a master joint roller, our code split the data into training and test sets, striking a perfect balance. 
- Then, it took Ridge Regression for a spin, creating a model that's as smooth and enjoyable as a good Sativa.
- Standardization was introduced, the kief of our data preprocessing, ensuring purity and consistency in our data as you would in your favorite bud.

**Data Preprocessing & CSV Creation**
- Our expert budtender, ahem, coder, tackled data

 preprocessing with finesse, and then packed our finely processed data into a CSV file, as perfectly rolled as a blunt ready for the weekend.

**Project Status Update**
- And, we wrapped up with another project status update, a vibrant narrative full of heady insights and intoxicating revelations, much like a philosophical discussion during a late-night sesh. 

There you have it, folks. As we venture into another day, remember, it's all about enjoying the journey, one line of code (and one puff) at a time.

**Changelog - "High" Level Overview, May 26th, 2023**

**Database Connection and Data Extraction**
- Our brilliant and smart-as-a-whip main character, the code, hooked up with SQLite, resulting in a sweet connection, much like a perfect drag on some Northern Lights. Cue the music; data is flowing, and boy, it's looking like a fresh pack of OG Kush. 

**Data Exploration and Preprocessing**
- Then, the code took a wild trip exploring and cleaning this glorious data, quite like cleaning a grinder. It looked at 'round_score', the 'sg' columns, 'round_num', 'course_par', and some groovy percentage columns, all while keeping things as chill as a GSC strain. 
- Then it whipped up a normalization joint for the strokes gained data, ensuring a balanced high all round.
- Using chi-square tests like a boss, the code selected features for round scores. It also sorted out the 'fin_text' column, clearing out the unwanted 'T' like it's a stem in your stash. 

**Dimensionality Reduction**
- The code embarked on a journey to reduce dimensionality (not the kind you experience with a potent sativa), replacing the bad trip of missing values, and introducing PCA like a pro. It then rolled up a new DataFrame filled with reduced features, smooth and ready to burn.

**Feature Engineering & Algorithm Development**
- It also set the stage for some epic feature engineering and algorithm development. Imagine it as planting the seeds for future growth; good times are a-comin'. 

**Prediction & Evaluation**
- The code even laid the groundwork for future prediction and evaluation, like a seasoned stoner pre-rolling for later.

**Main Execution Flow**
- The grand finale? A psychedelic journey through the main execution flow, with our star functions lighting up sequentially. Think of it as a perfect sesh, from the first spark to the mellowing out, it’s pure bliss. 

With our high-IQ code (I mean, it's sky-high, in the stratosphere, folks), we've reached new heights in preparing our data for machine learning, and setting up the foundations for future algorithm development and prediction tasks. Pass the munchies, we've earned them!

**Captain's Log - 28th Day of May, Year of Our Lord 2023**

**An Account of Our Calculations and Instruments:**
- The maps and charts are further enriched by our learned scribe's toils on the 'feature_engineering' function, giving us insights into such matters as average round scores, variance of round scores, and an overall reckoning of skill.
- A novel device, the 'create_model' function, has been fashioned, enabling the construction of a curious contraption called a neural network.

**Unfolding of Our Strategy and Training:**
- The 'algorithm_development' function has been bettered. It allows for the refinement of data, the training of our model, and the fine-tuning of a curious form of arithmetic called logistic regression.
- Our navigator has employed a technique of exhaustive search, known as grid search, to discover the most favourable conditions for our logistic regression.

**A Chronicle of Our Explorations and Prognostications:**
- The 'get_player_data' function now skillfully obtains the rolls of the players from the mysterious Data Golf Logbook.
- The 'predict_future_tournaments' function, akin to a fortune teller's crystal ball, now uses our trained model to foresee outcomes of future tourneys.

**A Reckoning of Our Model's Virtue:**
The 'analyze_and_evaluate_predictions' function has been fashioned to provide us with a reckoning of our model's skill and precision in terms of the Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R2 score.

**Maintenance and Troubles of Our Ship and Tools:**
Encountered a foul gust with the 'tensorflow' instrument. We suspect it may need some manner of adjustment or a new setup entirely.
Discussed the manner of restarting the Python – a creature not of nature, but a tool of our trade – in different environments, as a means to possibly mend the 'tensorflow' issue.

**The Entire Voyage:**
The entire course of our journey – the main execution flow of the program – is laid out from beginning to end. We've prepared for ill winds, with safeguards in place to manage any mishaps that may occur.

**Next Orders:**

- Address the issue with the 'tensorflow' instrument, possibly through adjustments or new setup.
- Further develop and fine-tune our predictive model based on the performance data of the players we've gathered.
- Make sure all parts of our journey are working together as expected.
- Make necessary course corrections based on what we learn from our maiden voyage.
- Look to include additional features or improvements based on the results of our voyage and the needs of our mission.

**Changelog - June 1, 2023 - "The Pride Edition"**

- Rainbow Refactoring: The code is celebrating Pride too! We took a hard look at ourselves, and just like discovering your true self, we found that some parts of our code could use a little touch-up. We embraced our inner Pythonista, got sassy with our functions and list comprehensions, and came out more beautiful than ever. Honey, that code is now a stunner!

- Prideful Processing: Our data's got all kinds of characters, and we love them all. But sometimes, sweetie, numbers have to be numbers. We had a bit of a coming out moment with our numeric columns, making sure they knew it was okay to be themselves (and numeric). We handled those special non-numeric entries like a drag queen handles a heckler – with style and grace.

- Dazzling Data: We talked about our 'tour_name' column – it's just like us, unique and diverse. We recognized that the fabulousness of different tours (like our fabulous friends PGA and Korn Ferry Tour) might carry different weights. And you bet we'll consider that in our model. We're all about inclusivity here!
Modeling and Voguing: We served some serious data science realness with our pipeline and GridSearchCV. There was a minor trip on the runway with a missing ColumnTransformer, but we sashayed away from that error by making sure our libraries strutted their stuff at the top of our script.

- Performance Extravaganza: We gave you the tea on the time it might take for our GridSearchCV performance to finish. A diva needs time to prepare, darling, and so does our model. Enough time for you to vogue, grab a drink, and maybe even take a disco nap!

- Serving Progress Realness: You wanted to know what's going on backstage during the GridSearchCV performance, and we served it hot! Just set the verbose level to 2 or 3, and you'll see all the drama unfolding, step by step.

- StratifiedKFold Eleganza: We had a minor fashion emergency with our StratifiedKFold warning – but we took it in stride. A wise queen once said, "Don't be a drag, just be a queen", and we might have a few less queens than we thought in our 'y' variable. But we'll handle that with all the fierceness we can muster!
That's the wrap, honey! This Pride edition was all about being true to ourselves and making sure our code was just as fabulous as we are. Let's sashay into the rest of Pride month with all the sass, flamboyance, and pride we can muster!

## Changelog - June 2, 2023

Dude, so check this, we've had a totally electrifying day. It was like being on the mainstage, with lights flashing and the crowd pulsing.

#### Database Handling Upgrade
We cranked up the volume on our Python DB script, making it more like a turntable where you can mix and match your beats. We gave it the ability to handle different sports like pgatour, nfl, ncaaf, ncaab, and horse racing, man. It's like a full-blown festival line-up! All of this while keeping things tight with unique data checks.

#### Incremental Learning
We added incremental learning to the party. It's like adding a new track to your set without stopping the music. We store the model after each training session, so when the next track (or data) comes in, we don't have to start from scratch.

#### GridSearchCV
We've been spinning the decks with GridSearchCV, tweaking parameters like a DJ tweaking the mixer. Added a couple of features there - parallel execution, verbosity settings, and duration predictions. 

#### SQL Optimization
Our SQL script was like a track that needed some remixing. We made sure the script doesn't waste time on data that's already there, kind of like not playing the same track twice, ya know?

#### Horse Racing Module
We've planned a new act in our festival - a horse racing module! We're gonna pull data from PDFs, like pulling sounds from a synth, and then jam it into SQLite tables.

#### Progress so Far
Like a day at a festival, we've been on a rollercoaster ride, man. Python scripts were refactored, GridSearchCV was fine-tuned, and SQL was optimized. But hey, the party ain't stopping yet!

Alright, gotta bounce now. Gonna ride this energy into the next set. Until the next update, keep the vibes high and the beats louder!

## Changelog - June 7, 2023

Well, well, well, if it ain't another day in paradise. Life is a dancefloor, and code is the beat, right? The rhythm just keeps pulsing, and the music just keeps playing. This is what went down today.

#### Data Transformation Remix
Like a DJ flipping the beat, we revamped our data handling routine. We conjured up some pivot tables for columns that were throwin' multiple rows at us, while keeping the static ones chill. It was like dropping a new rhythm while keeping the groove steady, man.

#### Missing Data? Not on My Watch!
We had some missing data in the house. Can't have that killing the vibe, can we? So we introduced IterativeImputer to the party. This dude filled in the gaps in the numeric data, letting the beat flow uninterrupted. 

#### Tuning the Sound System
Before diving into the groove, we had to make sure our data was hitting the right notes. We split our data into training and testing sets and tuned the frequencies with standard scaling. It was like getting the speakers and the mixer ready for the big show.

#### The DJ Lineup
We brought out the big guns - six models ready to rock the crowd. Linear Regression, Lasso Regression, Ridge Regression, Decision Tree, Random Forest, and Gradient Boosting - they all took turns at the decks. They gave us the tunes, and we measured the vibes with MAE, MSE, RMSE, MAPE, and R2 Score.

#### Save the Best for Later
Sometimes you play a set that's so lit, you gotta save it for later. That's why we used joblib to save our models. Now they can jump back on the decks anytime, without any prep. It's like having your playlist ready to go, anytime, anywhere.

#### The Afterparty
After a day of getting down with data and making models dance, I'm spent, man. But, looking at the refined data, the trained models, and the saved performances, it's been a day well grooved. Now, if only I could get some peace and quiet...

Well, gotta take care of those yapping furballs. Until the next update, keep your code clean and your beats dirty!

## Changelog - June 8, 2023

Just another "productive" day in the cozy confines of my basement. Let's dive into the rabbit hole of what's new:

#### Golf Get Files Method Overhaul
Revamped the "golf-get-files" method - a real renovation job, like turning a moldy basement into a 5-star man cave. We've got it pulling all the data we need now and parsing it correctly. A classy upgrade, kind of like swapping out your grandma's dusty old furniture for a sleek leather recliner and a pool table.

#### Pivot Table Magic
Conjured up some pivot table magic to transform and clean our data, like that time you discovered you could use a pizza box as a makeshift coffee table (don't judge, it's resourceful).

#### Data Imputation
Stuffed in missing values using a model-based imputation method. It's like filling up the holes in your life, or in this case, our data, with model predictions.

#### Model Feature Engineering
Brought some order to our features, and by order, I mean creating enough combinations to make a mathematician break out in cold sweat. Kind of like deciding which snacks to hoard when you're locked in your basement for days.

#### GridSearchCV Tuning
The real pot of gold was tuning our RandomForest model using GridSearchCV - the machine learning equivalent of endlessly tweaking your gaming settings for that sweet spot of performance and graphics.

#### Boostrap Resampling
Last but definitely not least, we got into some bootstrap resampling. Like when you've played a video game so many times you start seeing patterns, this lets us make pretty accurate guesses.

#### Progress So Far
Made some headway in the trenches of data science, and by that, I mean I did stuff while avoiding human interaction. Golf-get-files method revamped, data pivoted and imputed, model features engineered, GridSearchCV tuned, and bootstrap resampling in the bag. Time for a pizza break.

Catch ya on the flip side, or not, I could just live in this basement forever.

# Changelog: June 8, 2023

### Ahoy matey, the day was as long as a coiled python!

## Git Stuff

- The day started with the complex dance of git. Cloning, overwriting, all those boogie moves. You know the drill. But we tangled up a bit with syntax issues. Thanks to our trusty sidekick Python 3.x, we set things straight.

## The Great Python Migration

- Next, we wrangled our Python version, upgrading from the old-school Python 2.x to the hip Python 3.x. The old serpent didn't want to slither away quietly, but eventually we got our shiny new Python installed.

## Revving up the Raspberry Pi

- Then we moved on to our Raspberry Pi, a tiny but mighty beast. We tweaked and tuned, made it purr, then roar, optimizing it to utilize 75% of its resources for Python execution. We set up priority for Python execution, that little Pi never knew what hit it!

## Python Scripts & Data Wrangling

- We plunged into the depth of some Python scripts, optimizing and adding verbosity to a few of 'em dealing with data processing and machine learning. 

- Swung from Python pandas library to scikit-learn like a true jungle explorer, refining our ML pipeline for a golf dataset. The process was akin to swinging through the forest, replacing NaNs, one-hot encoding, and even running a GridSearch for the best RandomForest parameters. 

- Worked on some verbose print statements for a CSV data processing script. You know, because talkative scripts are friendlier.

## UI Shenanigans

- You asked for a UI for Python scripts? Mate, we talked about that too! We discussed Flask, Tkinter and PyQT. All those shiny buttons and sliders to play with!

## Performance Monitoring

- Finally, we rounded off with some Pi monitoring. Discussed Glances - a handy tool to check if our dear Pi is panting or chuckling at our computational demands.

And that's all folks! I'm off to get some grub and rest these tired eyes. It's been a real python of a day!