# Failure analysis — LSTM vs Transformer (seed 42, original-order test)

Test examples: 7,600.  Both correct: 6,782.  Both wrong: 380.  LSTM-only wrong: 239.  Transformer-only wrong: 199.


## Both models wrong (hard / ambiguous examples)  (380 total)

| # | text | true | LSTM pred (conf) | Transf. pred (conf) | trunc |
|---|------|------|------------------|---------------------|-------|
| 1 | Bryant Makes First Appearance at Trial (AP) AP - NBA star Kobe Bryant arrived at his sexual assault trial Monday as attorneys in the case w… | Sci/Tech | Sports (1.00) | Sports (1.00) |  |
| 2 | Nepal blockade 'blow to tourism' Nepal tour operators say tourists cancelled millions of dollars of bookings due to the rebel blockade of K… | Business | World (1.00) | World (1.00) |  |
| 3 | Indonesian diplomats asked to help improve RI #39;s bad image JAKARTA (Antara): President Susilo Yudhoyono asked Indonesian diplomats on Mo… | Business | World (1.00) | World (1.00) |  |
| 4 | Mars water tops science honours The discovery that salty, acidic water once flowed across the surface of Mars has topped a list of the 10 k… | World | Sci/Tech (1.00) | Sci/Tech (1.00) |  |
| 5 | Another homicide in Holland It is a sad day. In what seems to be another politically inspired homicide in Holland, Dutch filmmaker, and con… | Sci/Tech | World (1.00) | World (1.00) |  |
| 6 | Greek weightlifter awaits verdict Greek weightlifter Leonidas Sampanis will find out on Sunday if he is to be stripped of his medal. | World | Sports (1.00) | Sports (1.00) |  |

## Only the LSTM is wrong (Transformer correct)  (239 total)

| # | text | true | LSTM pred (conf) | Transf. pred (conf) | trunc |
|---|------|------|------------------|---------------------|-------|
| 1 | Med school move delayed to 2007 The MSU College of Human Medicine won #39;t be relocated to Grand Rapids until at least 2007, and could cos… | Business | Sci/Tech (1.00) | Business (0.53) |  |
| 2 | Racing in an Evening Gown (Forbes.com) Forbes.com - Not every driver was dressed formally for the start of this year's Bullrun, a road rall… | Business | Sports (1.00) | Business (0.79) |  |
| 3 | Argentina Beats U.S. Men's Basketball Team Argentina defeated the United States team of National Basketball Association stars 89-81 here Fr… | World | Sports (0.99) | World (0.70) |  |
| 4 | Great White Shark Loses Monitor Tag (AP) AP - A great white shark that was tagged with a data-gathering device in shallow waters off Cape C… | Sci/Tech | World (0.99) | Sci/Tech (0.52) |  |
| 5 | Security scare as intruder dives in A CANADIAN husband #39;s love for his wife has led to a tightening of security at all Olympic venues in… | Sports | World (0.99) | Sports (0.92) |  |
| 6 | Chirac: Europe Can Do More in Science Race (AP) AP - A European laboratory that was the birthplace of the World Wide Web and home of Nobel … | Sci/Tech | World (0.99) | Sci/Tech (0.97) |  |

## Only the Transformer is wrong (LSTM correct)  (199 total)

| # | text | true | LSTM pred (conf) | Transf. pred (conf) | trunc |
|---|------|------|------------------|---------------------|-------|
| 1 | At Last, Success on the Road for Lions The Detroit Lions went three full seasons without winning an away game, setting an NFL record for ro… | World | World (0.50) | Sports (1.00) |  |
| 2 | Meditation Practice Helping Arthritis Patients By ALEX DOMINGUEZ BALTIMORE (AP) -- Dalia Isicoff knows pain. A lifelong sufferer of rheumat… | Sci/Tech | Sci/Tech (0.99) | Sports (1.00) |  |
| 3 | Maddux Wins No. 302, Baker Wins No. 1,000 Greg Maddux pitched the Chicago Cubs into the lead in the NL wild-card race and gave Dusty Baker … | World | World (0.76) | Sports (1.00) |  |
| 4 | PRESS START FOR NOSTALGIA Like Led Zeppelin #39;s #39; #39;Stairway to Heaven #39; #39; and Lynyrd Skynyrd #39;s #39; #39;Freebird, #39; #3… | Sci/Tech | Sci/Tech (0.95) | Sports (1.00) |  |
| 5 | The Bag of Aeolus quot;Aeolus was keeper of the winds. He gave a bag of evil winds to Odysseus, instructing him to keep it closed while a g… | World | World (0.74) | Sports (0.99) |  |
| 6 | Munro, Morris Face Off in NLCS Game 2 ST. LOUIS - The Houston Astros put their hopes in a pitcher untested in the postseason when they give… | World | World (0.97) | Sports (0.99) |  |

## Part 2 — Type A: correct on original order, wrong after FULL shuffle

(Examples that genuinely needed word order.)  LSTM: 286 such examples.  Transformer: 160.


### LSTM — original-correct → full-shuffle-wrong

| # | text | true | wrong pred under shuffle (conf) |
|---|------|------|---------------------------------|
| 1 | Sports in brief He yelped after his second drive. His knees buckled after making contact on the sixth tee. . (See photo at left.) He stoppe… | Sports | Sci/Tech (1.00) |
| 2 | Catalina Foxes Back After Near Extinction (AP) AP - A unique subspecies of fox that is about the size of a house cat is back from the brink… | Sci/Tech | World (1.00) |
| 3 | Flying Cars Reportedly Still Decades Away (AP) AP - It's a frustrated commuter's escapist fantasy: literally lifting your car out of a clog… | Sci/Tech | World (0.99) |
| 4 | Dodgers Nip Giants 3-2 in Crucial Series SAN FRANCISCO - Shawn Green can sit out Saturday knowing he was a huge help to the Dodgers during … | World | Sports (0.99) |
| 5 | At Last, Success on the Road for Lions The Detroit Lions went three full seasons without winning an away game, setting an NFL record for ro… | World | Sports (0.99) |
| 6 | Bea Arthur for President Bea Arthur sparked a security scare at Logan Airport in Boston this week when she tried to board a Cape Air flight… | Sci/Tech | World (0.99) |

### Transformer — original-correct → full-shuffle-wrong

| # | text | true | wrong pred under shuffle (conf) |
|---|------|------|---------------------------------|
| 1 | Observers insist: no proof of fraud in Venezuelan referendum. Independent observers confirmed that the random auditing of results from the … | Business | World (0.99) |
| 2 | Post-Olympic Greece tightens purse, sells family silver to fill budget holes (AFP) AFP - Squeezed by a swelling public deficit and debt fol… | Business | World (0.99) |
| 3 | The Shockwaves of Sumatra The Indian Ocean earthquake of December 2004 produced a shockwave that created tsunamis all across the Indian Oce… | Sci/Tech | World (0.98) |
| 4 | Strikes at London airports London - A 48-hour strike by aircraft refuellers at London Heathrow airport got under way on Friday, with baggag… | Business | World (0.97) |
| 5 | Link Between Migraine, Endometriosis Found There #39;s evidence of a possible link between endometriosis and migraine, says an Italian stud… | Sci/Tech | World (0.96) |
| 6 | Airport Staff Walk-Out Fails to Disrupt Flights A strike by hundreds of baggage handlers and maintenance workers at Gatwick Airport failed … | Business | World (0.96) |
