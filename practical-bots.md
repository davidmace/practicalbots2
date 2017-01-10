#About the Author
David has worked on NLP algorithms for IBM Watson, Facebook's ad targeting, and Facebook's M personal assistant.

#Introduction
The goal of this mini-book is to teach a software developer how to write a bot engine. This is not a deep dive into the theory of natural language processing. Below I detail an opinionated machine learning architecture and provide all of the code you'll need to start writing your own bots.

#Table of Contents

1. Understanding the Problem and Data
  A. What is a Bot?
  B. Our Engine's Architecture
  C. Data You Expect vs Data You Get
  D. Heavy Tail of Edge Cases
2. Extracting Parameters
  A. Emails
  B. URLs
  C. Phone Numbers
  D. Numbers
  E. Datetimes
  F. Names, Locations
3. Word Correction
  A. Spelling Correction
  B. Colloquialism Correction
4. Extracting Custom Entities
  A. Simple Parameter Recognition
  B. Efficient Spellchecking for Large Lists of Custom Entities
5. Text Classification
  A. Grouping Training Data
  B. Convolutional Neural Network Model
6. State Management


#I. Understanding the Problem and Data

##A. What is a Bot Engine?

A bot converts a user's input text into a structured format, calls APIs to get and update background variables, then responds with a filled-in template. First look at the examples below then we'll break this down in more depth in the next section.

####Example 1
**User:** Find me a place to buy a Boston Terrier near Boston.
**Background API Call:** locallookup( location=Boston, item=Boston_Terrier) -> (id=LOC/125667, name=Jim's_Dog_Shelter)
**Bot:** How about this: Jim's Dog Shelter.
**User:** Thanks. Can you check if they're open tomorrow?
**Background Function Call:** is_open( id=ENT/125667, day=1/2/2017) -> True
**Bot:** Yes!

####Example 2
**User:** How many days until my package arrives?
**Background Function Call:** arrival_date(user_id=USER/13578) -> Error: multiple_packages
**Bot:** Which order?
**User:** my Kindle
**Background Function Call:** arrival_date(user_id=USER/13578, specifier=Kindle) -> 1/4/17
**Bot:** It's expected on January 4th!


##B. Our Engine's Architecture

There are five major components of our engine.

####1. Parameter Extraction
In the examples above, "Boston", "Boston Terrier", and "Kindle" are parameters. They are specific values that we need to pass to our API calls. Before the spellchecking phase, we extract regex entities (phone emails, urls, phone numbers, numbers, and datetimes) and proper nouns (names and locations). We still need to correct the spelling of custom entities before extracting them, so we don't handle them in this phase.

####2. Word Correction
Users often misspell words and use colloquial language (ie. ty, plz, k). It's important that we correct these errors before our algorithm attempts to understand the syntax and extract custom class parameters (ie. colors, burger toppings).

####3. Custom Parameter Extraction
Often a bot needs to recognize custom classes: for example color parameters (ie. red, green, purple) or room prices. This section details a simple architecture for writing and managing these custom entity lists. I also walk through a clever technique to efficiently spellcheck massive (10k-1m length) lists of custom entities (ie. hotel names, book titles, rock bands).

####4. Text Classification
Each of the user's inputs correspond to a function call with parameters. Above the function calls include local_lookup, is_open, and arrival_date. We first group past requests by resulting API call. Then for a new user input, we predict the best match function call by comparing the new request's syntax to past request syntax in each of the groups. 

####5. State Management
Requests often depend on previous information in the conversation. Parameters are passed between function calls. In this section, I detail a technique for managing this state information. This is definitely the most open-ended and application-specific section. Unlike with the components above, you may have to fiddle with these techniques to get it working on your use case.

##C. Data You Expect vs Data You Get

As a developer, you know it's hard to predict how users will interact with your UI. Natural language inputs have many more degrees of freedom than traditional UI inputs (ie. click, scroll, etc). It's extremely difficult to wrap your head around the types and complexity of your user's requests without actually digging into the data.

Your data will surely present its own challenges, but below are a few real-world examples of common complexities your bots will have to deal with.

####Example 1: Spelling and Grammar
**My Expected Input:** I would like to send apples Monday night after 3pm.
**User Input:** hi sir please i like to deliver apple monday night aftr 3 PM
**Analysis:** Our system needs to perform when faced with spelling and grammar errors. In one of my normal datasets from the United States, user inputs contain incorrect spelling ~30% of the time and incorrect grammar ~20% of the time. In one of my normal English datasets from Southeast Asia, these numbers jump to ~40% spelling errors and ~40% grammar errors.

####Example 2: Colloquialism
**My Expected Input:** May I please have the usual arrival time for tomorrow.
**User Input:** may i plz know expected arrival tym for tom
**Analysis:** Users often use shorthand or colloquial phrases like "tom" for "tomorrow" or "ty" for "thank you". Also language is very imprecise. Without context, it's difficult to know if the user is talking about a person "Tom" or is using shorthand for "tomorrow".

####Example 3: Complexity
**My Expected Input:** I am having trouble submitting my order.
**User Input:** I cant submit my order. what should I do? Shipping is selected but I get an error saying "please select a shipping option".
**Analysis:** Users often give you far more detail than you'd like. In this case, it's difficult to decipher whether to fall back on a stock "email support" answer or more specific answers on "shipping options".

####Example 4: Constraints
**My Expected Input:** Please find me a local restaurant in Hawaii.
**User Input:** Im looking for a local restaurant. We are in Hawaii. I'd prefer a place that serves salmon.
**Analysis:** Users have all sorts of weird constraints. First of all, it's difficult to linguistically recognize all of the odd conditions: "that serves salmon", "that costs less than $$", "within a mile of my location", etc. Even more importantly, users are going to surprise you with new types of constraints that your system has never seen (ie. "find me a hotel that's within walking distance of the beach and a grocery store").

####Example 5: Complexity++
**My Expected Input:** How does my son sign up for your school's classes?
**User Input:** My son is a student at Ohio State. He is a rising senior. I would like to enroll him in summer courses at your institution. Does he just sign up online, or should he go to an info session, or do I need to chat further with you?
**Analysis:** You might be able to answer really long and complex inputs with a stock answer. However, you're probably better off guiding your users to enter concise and direct inputs rather than trying to improve your technical solution. 

####Example 6: Answers Requiring Reasoning
**Previous Bot Output:** Are you an employee of the company or a prospect?
**My Expected Input:** employee
**User Input:** I have worked for you guys for 20 years.
**Analysis:** A human knows logically that if the user "works for you", then they're your employee. A computer does not. Algorithms are good at classifying phrase structures they've seen before, not reasoning out relationships.

####Example 7: Mixing Answers and Questions
**Previous Bot Output:** Which city would you like to book the hotel in?
**My Expected Input:** Lisbon.
**User Input:** Lisbon, can you make sure it's pet friendly
**Analysis:** Users mix answers and new requests in a single response. A good system needs to first store "Lisbon" as the previous question's response then process the follow-up request "can you make sure it's pet friendly". Splitting inputs into multiple requests/answers is a very hard unsolved task.

####Example 8: Missing Context
**Previous Bot Output:** Your book order has shipped.
**My Expected Input:** When is my book order arriving?
**User Input:** When is it getting here?
**Analysis:** New requests have to be taken in the context of past inputs. Our system needs to understand that "it" corresponds to the book order mentioned in the previous bot output.



##D. Heavy Tail of Edge Cases

Natural language data is generally heavy-tailed. This means that we have to account for lots and lots of esoteric edge cases. 

The graph below is the distribution of phone number formats from one of my real-world datasets. Phone number formats (ie. (111) 111 1111 and 111-111-1111) are much more consistent than other natural language tasks like "ways of saying I need X". However even on this simple task, 16 different formats appear in my sample of 30 phone numbers. Handling only the top 5 formats would miss ~50% of cases (ie. +11 111 111-1111 ext. 11). 

Let this be a reminder of the complexity of your users' inputs.

IMAGE HERE
![enter image description here](https://drive.google.com/file/d/0B5T-qsU04uQhWWRFakhVaGc1NFk/view)


#II. Extracting Parameters


## A. Emails

Email extraction is easy because the format <kbd>x@y.z</kbd> rarely if ever picks up false positives.

Here's the regex:
```
\b[\w\.\-!#$%&\'*\+\/=\?\^_`{\|}~]+@[\w\.\-]+\.[\w\-]+\b
```

Here's the code in python.
```
import re
s = 'my email is a@b.org.'
p = re.compile(r'\b[\w]+@[\w]+\.[\w]+\b')
print p.findall(s)
```

And some simple test cases:

```
a924-g@gmail.com, true
david-c.mace@hot-mail.com, true
send @ 5pm.tomorrow, false

```

> <i class="icon-info"></i>**Extra Note:** 
> If internationalization is a big concern, you should account for unicode characters (ie. david本@abc.com). The regex above works if you first convert your string to [Punycode](https://en.wikipedia.org/wiki/Punycode).

##B. URLs

URL extraction is more complicated because we need to recognize all of the following formats:
```
https://www.google.com
www.google.com
google.xyz
google.xyz/dogs-cats?dogs=500
david.google.xyz:3000/dogs#33
google.xyz?i+like+turtles
```
  
And we need to ignore the following formats:
```
bye.done with trial
I am at Google.come to the park.
user@google.com
```

The best regex I’ve seen looks for the pattern <kbd>wx.yz</kbd> where w contains http(s):// and/or www , x contains valid characters for a domain name, y is in a list of valid domain endings, and z belongs to a set of valid characters for url parameters, port, etc.

Here's that regex (make sure to run in case-insensitive mode):
```
(?:https?\:\/\/)?[\w\-\.]+\.(?:'+'|'.join(top_level_domains)+')[\w\-\._~:/\?#\[\]@!\$&%\'\(\)\*\+,;=]*
```

And the python code.
```
with open('top-level-domains.txt','r') as f :
    top_level_domains = f.read().split('\n')
s = 'bye.done google.com http://david.google.xyz:3000/dogs'
p = re.compile(r'(?:https?\:\/\/)?[\w\-\.]+\.(?:'+'|'.join(top_level_domains)+')[\w\-\._~:/\?#\[\]@!\$&%\'\(\)\*\+,;=]*', re.I)
print p.findall(s) 
```

And here's my list of common url endings that has 98% coverage:
> <i class="icon-download"></i> [top-level-domains.txt](https://en.wikipedia.org/wiki/List_of_Internet_top-level_domains)

Also you should be careful not to recognize emails (david@google.com) as urls (google.com). To solve this problem, first extract emails then check that extracted urls are not substrings of your extracted emails.

> <i class="icon-info"></i>**Extra:** 
> I think it's overkill but if you need better than 98% url ending coverage, you can find the updated full list here: [wikipedia top level domains](https://en.wikipedia.org/wiki/List_of_Internet_top-level_domains)

> <i class="icon-info"></i>**Extra:** 
> Like emails, urls can contain unicode characters. For instance .рф is the top level domain for ~0.1% of websites. If internationalization is a big concern, the regex above works if you first convert your string to [Punycode](https://en.wikipedia.org/wiki/Punycode).

##C. Phone Numbers
Phone number extraction is more complex than you might imagine. If we want our technique to work on international numbers, we have to handle all of these cases:
```
+12-555-555-5555
555 555 5555
5555555
555.5555
555-5555 x.7
555-5555, Ext 8
555-5555 extension 3
(01 55) 5555 5555
0455/55.55.55
05555 555555-55
```

But not these:
```
4500
I have 100. 1000 are on the way
I have between 100-1000 berries
```
Without taking nearby words into account, it’s not possible to differentiate some phone numbers from normal numbers (ie. 5555555 or 100-1000). This is rare enough that I usually just mark any number with seven or more digits/dashes as a phone number.

Here's the regex (make sure to run as case-insensitive):
```
[\s\-\d\+\(\)\/\.]{7,}[\s\.,]*(?:x|ext|extension)?[\s\.,\d]*\d
```

And the python code.
```
s = '4567 abc +12-555-555-5555 abc 555 555 5555 abc 5555555 abc 555.5555 abc 555-5555 x.7 abc 555-5555, Ext 8 abc 555-5555 extension 3 abc (01 55) 5555 5555 abc 0455/55.55.55 abc 05555 abc 555555-55'
p = re.compile(r'[\s\-\d\+\(\)\/\.]{7,}[\s\.,]*(?:x|ext|extension)?[\s\.,\d]*\d', re.I)
print p.findall(s)
```

##D. Numbers

Extracting digit-based numbers is simple. However people often don't write numbers as digits. They say "five thousand" or "5k". Regex isn't the best move since we probably want to map 72 thousand -> 72000 rather than just recognizing that 72 thousand is a number.

Here are some examples of the weird cases we need to handle:

```
9000
9,000
9.0
1.5m
a million
four hundred thousand and twenty eight
72 thousand
one and a half
```

Also make sure to ignore numbers that we have previously identified as parts of phone numbers.

The files below specify written-out number parsing rules (ie. eighty two, 八十二, quatre-vingt-deux) in a format that our parser (process_numbers.py) can process. The examples below are rulesets for English, Mandarin, and French. If you need to support another language, all you have to do is write a new ruleset file in the specified format. The parser will do the rest of the work.

'N' specifies a number word, 'B' specifies a base word, and 'L' specifies a link word. All words after the colon are interchangeable ways of representing the integer or decimal value before the colon.

<br>
<i class="icon-file"></i> **numbers-english.txt**
```
N 0: zero
N 1: one
N 2: two
N 3: three
N 4: four
N 5: five
N 6: six
N 7: seven
N 8: eight
N 9: nine
N 10: ten
N 11: eleven
N 12: twelve
N 13: thirteen
N 14: fourteen
N 15: fifteen
N 16: sixteen
N 17: seventeen
N 18: eighteen
N 19: nineteen
N 20: twenty
N 30: thirty
N 40: forty
N 50: fifty
N 60: sixty
N 70: seventy
N 80: eighty
N 90: ninety
B 100: hundred
B 1000: thousand
B 1000000: million
B 1000000000: billion
B 0.5: halves half
B 0.33: thirds third
B 0.25: quarters fourths quarter fourth
B 0.2: fifths fifth
B 0.1: tenths tenth
L: and_a and
```

<br>
<i class="icon-file"></i> **numbers-mandarin.txt**
```
N 0:零
N 1:一
N 2:二
N 3:三
N 4:四
N 5:五
N 6:六
N 7:七
N 8:八
N 9: 九
B 10: 十
B 100:百
B 1000:千
B 10000:万
N 0.5: 一半 二分之一
N 0.33: 三分之一
N 0.25: 四分之一
N 0.2: 五分之一
N 0.1: 十分之一
```

Here is a ruleset for French:
> <i class="icon-download"></i> **French Ruleset:** https://github.com

And here is a link to the number parsing code. It handles the language-specific number formats above and non-language-specific number formats (ie. 5.2k, 678).
> <i class="icon-download"></i> **Number Parser Code:** https://github.com

<br>
> <i class=" icon-thumbs-down-alt"></i> **Limitation:** The number parser cannot handle number formats where the base is on the left of the number (ie. 十分之三 is 3/10 but the first three characters specify the base 1/10 and the last character specifies the number 3). Also because the parser is language-agnostic, it will recognize numbers formats that aren't allowed in a language (ie. two thousand twenty thirty three-> 2053). I have never seen this cause an issue in practice.


##E. Time Extraction

Times are the hardest entity to properly extract. Ideally we also want to map each time string to a numerical calendar value which we can query later. Here are some examples to show why it's so hard:

```
yesterday at 7:30am
second Wednesday of April
in 35 minutes
January 4th at 3pm
tomorrow at half past noon
```

Wit.AI’s Duckling has the highest accuracy of any library I've seen. It supports English, Spanish, French, Italian, and Chinese, which together cover 67% of online spoken language. More importantly it's written in a way (probabilistic models) that makes it easily extensible to further languages.

Duckling additionally offers number, email, url, etc parsing; however, I’ve found regex parsers to work better on all entities except time (as of Sept 2016). 

Duckling is written in clojure so I wrapped it in a simple service. Duckling does a non-negligible amount of work so you'll probably want to easily scale it up/down later (which is easiest if it's a separate service from the main logic code). 

> <i class="icon-download"></i> **Duckling Server Code:** https://github.com


##F. People and Locations
Here's an example of extracting names and locations from text.
```
  1. hi im sakthi -> {sakthi:name}
  2. minh is in south dakota -> {minh:name, south_dakota:location}
  3. san jose is better than sf -> {san_jose:location, sf:location}
```
The simplest approach to recognizing names is to make a big list of names from census records, Wikipedia, etc. This is possible. However, it’s more scalable to use an established Named Entity Recognizer (NER) because its performance will improve year over year. It’s also hard to get name lists working well for foreign names (ie. John-Paul, Veeral, Vi)—there are so many odd possibilities that you’ll probably miss the rarest 10%.

I recommend Google’s NL API Entity Recognizer (https://cloud.google.com/natural-language/docs/) because it's fast, scalable, decently although not perfectly accurate, and will likely improve over time.

Sometimes stock NERs perform poorly on lowercase text (ie. sarthak went to bangalore). Since users rarely capitalize input text, this is a major problem. To improve performance without training your own model, you can manually capitalize non-common words. Below is a list of the most common non-entity words in English. Capitalize every word that isn't contained in this list.
```
  i went park in delhi yesterday -> i went park in Delhi yesterday
```

Common Word List:
> <i class="icon-download"></i> [common-non-entity-words.txt](http://www.google.com)

Here's some code to capitalize the rare words.
```
import nltk

with open('common-non-ent-words.txt','r') as f :
    words = f.read().split('\n')

def capitalize_rare_words(sentence) :
  tokens = nltk.word_tokenize(sentence)
  for i in range(len(tokens)) :
      if tokens[i] not in words :
          tokens[i] = tokens[i][0].capitalize() + tokens[i][1:]
  return tokens
```


#Preprocessing

##A. Spelling Correction

Hunspell is a high accuracy, easy to deploy spellchecking solution. It's built into Chrome and Safari among others, which means that it's likely to continue being supported at least in the near-intermediate term. It also supports a variety of languages, which is important for internationalization.

Hunspell overcorrects entity types like ordinals (ie. 4th) and proper nouns. This is why it was important to extract these parameters and hide the entity behind a special key "#n#" before this section. 

Below is a microservice that wraps the hunspell open source library. Hunspell does a considerable amount of work, which is why it makes sense to deploy it as a separate service (so you can easily scale it up/down later).

> <i class="icon-download"></i> **Hunspell Server Code:** https://github.com


##B. Colloquialism Correction

Spelling correctors don't pick up on Internet slang <kbd>thx->thanks</kbd> <kbd>kk->okay</kbd>. For example we might want to respond "No problem" to any form of "thanks" "thks" "thx" "ty". If we had all the data in the world, our algorithm would learn to recognize each of these responses separately. However in most cases we will never see at least one of these forms of "thanks" in the training data. Because of this, we need to normalize them to the same word.

Here is my list of English internet abbreviations and their full spellings:
> <i class="icon-download"></i> [internet-abbreviations.csv](github.com)

<br/>
> <i class="icon-info"></i>**Extra:** 
I only extracted colloquialisms for English. To generate a colloquialism list for other languages, here's a process that works well. Obtain a large amount (1m+ lines) of online conversation data (I used a day's worth of public Tweets). Eliminate every word that is present in a dictionary. Sort the remaining words by frequency. Manually go through the 500 words with highest frequency and extract all of the Internet abbreviations in the list.


#3. Extracting Custom Entities

##A. Simple Parameter Recognition

What if you want to extract custom parameters? Here are some examples for grocery delivery.
```
i want a bunch of bananas
plz deliver two bags of naval oranges
add a grapefruit
```

We'll worry later about recognizing qualifiers like "two bags" and "bunch". For now we're just trying to recognize singular and plural keywords (ie. oranges, grapefruit, bananas).

This is simple so I won't write out the code. Use the 'stem' function from nlp_helpers.py to first reduce plural/singular word forms to a common stemmed form. Then exact match keywords. Remember that multiple words can refer to the same concept (ie. brush, scrubber).

> <i class=" icon-thumbs-down-alt"></i> **Limitation:** Exact matching parameters isn't perfect, even after spelling correction. It seems at first glance like there might be some nice way to automatically recognize similar words as the same concept (for example by using word2vec distance or wordnet). In practice, these techniques produce far too many false positives and it's surprisingly tractable to quickly write out explicit lists that perform well.

##B. Efficient Spellchecking for Large Lists of Custom Entities

Hunspell only fixes the spelling of common words. If you want to recognize a large (10k+ line) list of proper nouns, you'll need to add more spell-checking logic.

Classic spellchecking algorithms (ie. edit distance) are far too slow if you're trying to recognize huge lists of words/phrases. Here's a clever technique to do quick custom spellchecking on large lists of candidates (source: EMNLP Popescu et al 2014).

> <i class="icon-download"></i> [large-custom-spellchecker.py](github.com)

The algorithm works by the following logic.

1. Assign each letter/character to a prime number in ascending order (ie. ' ':2 a:3, b:5, c:7, d:11...).
2. Reduce each of your parameter names to a number hash by multiplying together the prime numbers that correspond to each letter. (ie. adcc -> 3*11*7*7 -> 1617).
3. When we don't recognize a word/phrase in the user's input, get its number hash by the same method as above.
4. If there is a 1 or 2 character error in the user's entered word/phrase, it will be off from the actual entity's hash by a factor of either 1/a, a, 1/a/b, ab, or a/b where a,b∊[2,3,5,7,11...]. Take all candidate spelling matches with hashes that are off by any of the factors in this list. These are constant-time lookups.
6. The resulting candidate list is vastly reduced so we can do a more expensive edit-distance computation for each of the candidates. This will eliminate candidates that have closely aligned letters but in incorrect positions.


# 4. Text Classification (TODO)

##A. Dependency Parses (Probably Remove)

A dependency parse is a structured representation of a phrase or sentence. It tells us the relationships between words. For example the following sentence contains each of the dependencies below.
> I gave the dogs to Mary ->
> (root,root,gave)
> (nsubj,gave,I)
> ...

To see why this is useful, consider the example below. Imagine we want to know who the dogs were given to.

> I gave the dogs quickly to Lily
> I had given the dogs yesterday to Bill
> I will give the dogs from the park to Bob

The language is too complex to easily extract the relationship between the verb "give" and the people. However, the dependency parse has given us structured information so all we have to do is look for dependencies of the form <kbd>(iobj,give,ANSWER)</kbd>.

There are a few options for parsing. Stanford Parser is historically the most accurate but it did not scale well for my past projects and I wasted time trying to optimize its internal resources. Hosting your own Tensorflow-based Parsey Mcparseface as a microservice seems like a good idea in theory, but you incur a large technical cost to managing the scaling of this service. Additionally if you host your own parser, its accuracy will eventually lag behind public parsing APIs as they include new research results in the coming years. 

For these reasons, I recommend a paid parsing service—I’ve specifically had great luck with Google’s NL API. A parsing service is more expensive than raw cloud compute time, but probably not if you include the lost development time spent managing and debugging your own microservice.


><i class="icon-info"></i>**Extra:** If you’re feeling bold, here’s containerized code to run a TensorFlow-based simple parsing service that you can customize.
> <i class="icon-download"></i> [tf-parser](github.com)

><i class="icon-info"></i>**Extra:** TODO constituency parse link


##B. (Probably Remove)

You'll need data from humans acting as your bots. You'll need to manually curate this data. I have in the past spent countless hours trying to train a bot on raw customer support logs. Trust me. Don't do it. It's tedious but you'll need to semi-manually curate the logs to group the important phrases. I have included a 

For instance, you'll need to split single user responses into multiple requests.

Technique should work okay with only ~5 phrases in group and quite well with ~100 phrases in group.

- Take 1,2,3,4-grams from dependency parse and compare the new request to the distribution for each group in a way that leverages word2vec and grammar (some phrase2vec for each gram then some distribution comparison technique)

- constraints can either be recognized as stand-alone category if P is greatest or as add-on if P>threshold

Can you find me a restaurant gluten-free

gluten-free
food without gluten



