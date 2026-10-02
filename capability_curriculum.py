"""Authored, bounded English/task curriculum; no inference answer lookup."""
import hashlib
import re


def split_for(group):
    bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 20
    return "test" if bucket == 0 else "validation" if bucket == 1 else "train"


def normalize(text):
    return re.sub(r"\s+", " ", text).strip().casefold().rstrip(".!?")


DEFINITIONS = [
    ("a computer", "A computer is an electronic machine that processes information and runs programs."),
    ("gravity", "Gravity is the attraction between objects with mass. It keeps us on Earth and planets in orbit."),
    ("a CPU", "A CPU is a computer's central processing unit. It executes instructions and handles many general tasks."),
    ("a GPU", "A GPU performs many calculations in parallel. It is useful for graphics and training neural networks."),
    ("RAM", "RAM temporarily stores data that a computer is using. Its contents are usually lost when power is turned off."),
    ("an SSD", "An SSD stores files using flash memory. It keeps data even when the computer is switched off."),
    ("a Python function", "A Python function is a reusable block of code that performs a task. It can take inputs and return a result."),
    ("a variable", "A variable is a name that refers to a value in a program."),
    ("a loop", "A loop repeats a block of code. It can process items in a collection or run while a condition is true."),
    ("a Python list", "A Python list is an ordered, mutable collection of values. You can add, remove, and change its items."),
    ("a Python tuple", "A Python tuple is an ordered collection whose items cannot be replaced after the tuple is created."),
    ("a dictionary in Python", "A Python dictionary stores key-value pairs. You use a key to look up its associated value."),
    ("a string", "A string is a sequence of characters used to represent text."),
    ("a boolean", "A boolean has one of two values: true or false."),
    ("an integer", "An integer is a whole number, such as -2, 0, or 7."),
    ("an algorithm", "An algorithm is a sequence of steps for solving a problem or completing a task."),
    ("a bug in a program", "A bug is an error in a program that causes incorrect or unexpected behavior."),
    ("debugging", "Debugging means finding and fixing errors in a program."),
    ("a syntax error", "A syntax error occurs when code breaks the grammatical rules of a programming language."),
    ("an exception", "An exception signals a problem during program execution. Code can catch and handle some exceptions."),
    ("a unit test", "A unit test checks a small piece of code against an expected result."),
    ("recursion", "Recursion happens when a function calls itself. It needs a stopping condition to avoid continuing forever."),
    ("Python", "Python is a general-purpose programming language used for automation, web development, data analysis, and machine learning."),
    ("Java", "Java is a programming language commonly used for applications and server software. Java bytecode runs on a Java Virtual Machine."),
    ("JavaScript", "JavaScript is a programming language used to add behavior to web pages and to build other applications."),
    ("HTML", "HTML describes the structure of a web page using elements such as headings, paragraphs, and links."),
    ("CSS", "CSS controls the appearance of a web page, including colors, spacing, and layout."),
    ("a database", "A database stores organized information so that it can be searched, retrieved, and updated."),
    ("an API", "An API is an interface that lets software components communicate through defined requests and responses."),
    ("an operating system", "An operating system manages computer hardware and provides services that applications use."),
    ("Linux", "Linux is an operating system kernel. Linux distributions combine it with other software to provide a complete system."),
    ("a file", "A file is a named collection of data stored on a device."),
    ("a folder", "A folder organizes files and other folders."),
    ("the internet", "The internet is a worldwide network of interconnected computer networks."),
    ("a web browser", "A web browser is an application that retrieves and displays web pages."),
    ("a server", "A server provides data or services to other programs or devices over a connection."),
    ("a password", "A password is a secret used to help verify access to an account or system."),
    ("encryption", "Encryption transforms readable data into a protected form that requires the appropriate key to read."),
    ("machine learning", "Machine learning uses data to learn patterns that help a computer make predictions or perform tasks."),
    ("a neural network", "A neural network is a model with connected layers that transform inputs using learned weights."),
    ("a language model", "A language model learns patterns in text and predicts tokens. It can generate sentences but may make factual mistakes."),
    ("a tokenizer", "A tokenizer splits text into units called tokens and maps them to numbers that a model can process."),
    ("a model parameter", "A model parameter is a learned value, such as a weight, that affects the model's predictions."),
    ("training loss", "Training loss measures prediction error on training examples. Lower training loss alone does not prove better answers on new questions."),
    ("validation data", "Validation data is kept separate from training updates and helps measure how well a model generalizes."),
    ("overfitting", "Overfitting happens when a model learns training examples too closely and performs poorly on new examples."),
    ("gradient descent", "Gradient descent updates model parameters in a direction intended to reduce a loss function."),
    ("a learning rate", "The learning rate controls the size of parameter updates during training."),
    ("a batch", "A batch is a group of examples processed together during model training."),
    ("an epoch", "An epoch is one pass through the training dataset."),
    ("a checkpoint", "A checkpoint saves a model's state so it can be loaded later. Training checkpoints may also save optimizer state."),
    ("a context window", "A context window is the amount of text a language model can consider at one time, measured in tokens."),
    ("photosynthesis", "Photosynthesis lets plants use light energy to make sugars from water and carbon dioxide, releasing oxygen."),
    ("evaporation", "Evaporation is the change of a liquid into a gas at its surface."),
    ("condensation", "Condensation is the change of a gas into a liquid, such as water vapor forming droplets."),
    ("an atom", "An atom is a basic unit of ordinary matter. It has a nucleus surrounded by electrons."),
    ("a molecule", "A molecule consists of atoms bonded together."),
    ("energy", "Energy is the capacity to do work or cause change. It can take forms such as heat, motion, and light."),
    ("friction", "Friction is a force that opposes relative motion between surfaces in contact."),
    ("electric current", "Electric current is the flow of electric charge."),
    ("a battery", "A battery stores chemical energy and can convert it into electrical energy."),
    ("a magnet", "A magnet produces a magnetic field and can attract materials such as iron."),
    ("a planet", "A planet is a large body that orbits a star. Earth is a planet in the Solar System."),
    ("a star", "A star is a hot ball of plasma that produces energy through nuclear fusion. The Sun is a star."),
    ("the Moon", "The Moon is Earth's natural satellite. It reflects sunlight and orbits Earth."),
    ("a noun", "A noun names a person, place, thing, or idea."),
    ("a verb", "A verb expresses an action, event, or state."),
    ("an adjective", "An adjective describes a noun, as in a red ball or a tall tree."),
    ("an adverb", "An adverb can modify a verb, adjective, or another adverb, as in running quickly."),
    ("a sentence", "A sentence expresses a complete thought and normally begins with a capital letter and ends with punctuation."),
    ("a paragraph", "A paragraph is a group of related sentences that develop a main idea."),
    ("a fraction", "A fraction represents a number as one quantity divided by another, such as 1/2."),
    ("a prime number", "A prime number is an integer greater than 1 with exactly two positive divisors: 1 and itself."),
    ("an even number", "An even integer is divisible by 2 without a remainder."),
    ("an odd number", "An odd integer is not divisible by 2 without a remainder."),
    ("a triangle", "A triangle is a polygon with three sides and three angles."),
    ("a square", "A square has four equal sides and four right angles."),
    ("a circle", "A circle is the set of points in a plane at a fixed distance from its center."),
]

EXPLANATIONS = [
    ("Why do plants need sunlight?", "Plants use light energy for photosynthesis, which lets them make food from water and carbon dioxide."),
    ("Why does it rain?", "Water vapor condenses into droplets in clouds. When the droplets become heavy enough, they fall as rain."),
    ("Why is the sky blue?", "Air molecules scatter sunlight. They scatter blue light more strongly than red light, making the daytime sky look blue."),
    ("Why do we have day and night?", "Earth rotates. The side facing the Sun has daylight, while the side facing away has night."),
    ("Why should I test my code?", "Tests help find mistakes and check that changes preserve expected behavior."),
    ("How can I learn programming?", "Choose one language, learn its basics, and build small projects. Read error messages and practice solving simple problems."),
    ("Give me one tip to learn programming.", "Practice by building a small project and fixing the errors you encounter."),
    ("How can I improve my English sentences?", "Use a clear subject and verb, check the tense, and read the sentence aloud. Practice rewriting short sentences."),
    ("How can I improve a language model?", "Use clean and varied training data, check the training code, and compare answers on examples excluded from training."),
    ("Does a low training loss mean my model is intelligent?", "No. A model can memorize training data. Evaluate its answers on new questions to check understanding and accuracy."),
    ("What is the difference between a CPU and a GPU?", "A CPU handles varied general tasks. A GPU is designed to perform many similar calculations in parallel."),
    ("What is the difference between RAM and storage?", "RAM holds data currently in use and is usually temporary. Storage keeps files when the power is off."),
    ("What is the difference between a list and a tuple in Python?", "A list is mutable, so its items can be changed. A tuple is immutable, so its items cannot be replaced."),
    ("What is the difference between training and inference?", "Training updates a model's learned parameters. Inference uses those parameters to produce an output."),
    ("What is the difference between hardware and software?", "Hardware is the physical equipment. Software is the programs and instructions that run on it."),
    ("Can language models make mistakes?", "Yes. Language models can produce fluent but incorrect answers. Check important claims against reliable evidence."),
    ("Can you see my screen?", "I do not have screen access through this chat model."),
    ("What is the weather right now?", "I do not have live weather data. Check a weather service for your location."),
    ("What is my password?", "I do not know your password."),
    ("Who will win tomorrow's match?", "I cannot know the result of a future match."),
    ("Write one sentence about rain.", "Rain falls from the clouds and waters the plants."),
    ("Write one sentence about a dog.", "The friendly dog ran across the garden."),
    ("Write one sentence about learning.", "Learning a little every day helps us build new skills."),
    ("Write a short story about a lost cat.", "A small cat wandered away from home. A kind neighbor found it near a tree and helped it return to its family."),
    ("Help me.", "What would you like help with?"),
    ("Can you fix this?", "Please share the text or problem you want me to fix."),
    ("Explain it.", "What would you like me to explain?"),
]


def make_curriculum():
    rows = []
    def add(category, group, prompt, answer, split=None):
        rows.append(dict(category=category, group=group, prompt=prompt,
                         answer=answer, split=split or split_for(group), source="authored"))

    grammar_prefixes = ["Fix this sentence: ", "Correct the grammar: ", "Correct my English: "]
    def grammar(bad, good):
        group = "grammar:" + normalize(good)
        split = split_for(group)
        prefixes = grammar_prefixes if split == "train" else ["Please correct this sentence: "]
        for prefix in prefixes:
            add("grammar", group, prefix + bad, good, split)

    subjects = [("I", False, "am"), ("You", False, "are"), ("We", False, "are"),
                ("They", False, "are"), ("He", True, "is"), ("She", True, "is"),
                ("Tom", True, "is"), ("Mia", True, "is"), ("The boy", True, "is"),
                ("The girl", True, "is"), ("The children", False, "are")]
    verbs = [("go", "goes", "going", "to school every day"),
             ("walk", "walks", "walking", "to work"),
             ("read", "reads", "reading", "a book"),
             ("play", "plays", "playing", "football"),
             ("cook", "cooks", "cooking", "dinner"),
             ("write", "writes", "writing", "a letter"),
             ("watch", "watches", "watching", "a movie"),
             ("drink", "drinks", "drinking", "water"),
             ("eat", "eats", "eating", "breakfast"),
             ("learn", "learns", "learning", "English"),
             ("make", "makes", "making", "a cake"),
             ("help", "helps", "helping", "a friend"),
             ("wash", "washes", "washing", "the dishes"),
             ("visit", "visits", "visiting", "the park"),
             ("need", "needs", None, "more time"),
             ("like", "likes", None, "coffee")]
    for sub, singular, be in subjects:
        for base, third, ing, obj in verbs:
            grammar(f"{sub} {base if singular else third} {obj}.",
                    f"{sub} {third if singular else base} {obj}.")
            grammar(f"{sub} {'do not' if singular else 'does not'} {base} {obj}.",
                    f"{sub} {'does not' if singular else 'do not'} {base} {obj}.")
            if ing:
                grammar(f"{sub} {be} {base} {obj}.", f"{sub} {be} {ing} {obj}.")
        for state in ["happy", "tired", "ready", "late", "hungry", "at home", "in the garden", "reading books", "learning Python"]:
            grammar(f"{sub} {'are' if be in ('is','am') else 'is'} {state}.", f"{sub} {be} {state}.")
        for obj in ["a pen", "a book", "two pens", "three apples", "a red bag", "a new computer"]:
            grammar(f"{sub} {'have' if singular else 'has'} {obj}.", f"{sub} {'has' if singular else 'have'} {obj}.")
        for number in ["two", "three", "four", "five"]:
            for noun in ["pen", "book", "apple", "bag", "chair"]:
                grammar(f"{sub} {'have' if singular else 'has'} {number} {noun}.",
                        f"{sub} {'has' if singular else 'have'} {number} {noun}s.")

    for op in ["+", "-", "*"]:
        limit = 12 if op == "*" else 20
        for a in range(limit + 1):
            for b in range(limit + 1):
                if op == "-" and b > a:
                    continue
                value = a+b if op == "+" else a-b if op == "-" else a*b
                word = {"+":"plus", "-":"minus", "*":"times"}[op]
                # Commutative reversals share a split.
                nums = sorted((a,b)) if op in ("+","*") else (a,b)
                group = f"arithmetic:{op}:{nums[0]}:{nums[1]}"
                prompts = [f"What is {a} {op} {b}?", f"Calculate {a} {word} {b}.", f"Hi, what is {a} {word} {b}?"]
                if split_for(group) != "train":
                    prompts = [f"Please calculate {a} {word} {b}."]
                for prompt in prompts:
                    add("arithmetic",group,prompt,f"{value}.")
                if op == "+" and a > 1 and b > 1:
                    add("arithmetic", group, f"Sam has {a} apples and gets {b} more. How many apples does Sam have?", f"Sam has {value} apples.")
    for a in range(1, 26):
        for b in range(a + 1, min(31, a + 6)):
            for first, second in [(a,b),(b,a)]:
                group=f"comparison:{a}:{b}"
                add("comparison",group,f"Which is larger, {first} or {second}?",f"{b} is larger.")
                add("comparison",group,f"Which is smaller, {first} or {second}?",f"{a} is smaller.")

    colors=["red","blue","green","yellow","black","white"]
    names=["Mia","Tom","Sam","Lily","Anna","Ben","Ravi","Sara"]
    objects=["bag","hat","ball","bike","cup","box"]
    for ni,name in enumerate(names):
        other=names[(ni+1)%len(names)]
        for ci,color in enumerate(colors):
            for obj in objects:
                group=f"reading:{name}:{color}:{obj}"
                add("reading",group,f"{name} has a {color} {obj}. What color is {name}'s {obj}?",f"{name}'s {obj} is {color}.")
                add("reading",group,f"{name} has a {color} {obj}. {other} has a {colors[(ci+2)%6]} {obj}. What color is {name}'s {obj}?",f"{name}'s {obj} is {color}.")
                add("reading",group,f"{name} has a {color} {obj}. Who has the {color} {obj}?",f"{name} has the {color} {obj}.")
    for term, answer in DEFINITIONS:
        group="definition:"+term
        for prompt in [f"What is {term}?",f"Explain {term}.",f"Tell me about {term}.",f"What does {term} mean?",f"Briefly explain {term}."]:
            add("explanation",group,prompt,answer,"train")
        add("explanation",group,f"Can you explain {term} in simple words?",answer,"validation")
        add("explanation",group,f"Give a short explanation of {term}.",answer,"test")
    for question, answer in EXPLANATIONS:
        group="explanation:"+normalize(question)
        for prefix in ["", "Hi, ", "Please answer briefly: "]:
            add("explanation",group,prefix+question,answer,"train")
        add("explanation",group,"Please help me understand: "+question,answer,"validation")
        add("explanation",group,"Answer in simple English: "+question,answer,"test")
    greetings=["hi","Hi!","HI","hello","Hello!","hey","Hey!","hiya","hi bro","hello there","good morning","good evening","hlo","hii"]
    for q in greetings:
        add("chat","greeting",q,"Hi! How can I help you?","train")
    for q in ["hey there", "hello bro"]:
        add("chat","greeting",q,"Hi! How can I help you?","validation")
    for q in ["Hi there!","hey bro","Good afternoon!"]:
        add("chat","greeting",q,"Hi! How can I help you?","test")
    for qs,a in [(["Who are you?","What is your name?"],"I am CrachoLM, a small experimental language model."),
                 (["How are you?","How are you doing?"],"I am ready to help. How are you?"),
                 (["thanks","thank you","thanks bro"],"You are welcome!"),
                 (["bye","goodbye","see you later"],"Goodbye! Have a good day.")]:
        for q in qs:
            add("chat","chat:"+normalize(q),q,a,"train")
    # Teach preservation of extra information, not just the common short answer.
    originals = [r for r in rows if r['category']=='grammar']
    for r in originals:
        if r['prompt'].startswith('Fix this sentence: '):
            bad=r['prompt'].split(': ',1)[1].rstrip('.')
            good=r['answer'].rstrip('.')
            if not any(word in good for word in ['every day',' at night']):
                for suffix in ['every day','in the morning','after school','on Monday']:
                    grammar(bad+' '+suffix+'.',good+' '+suffix+'.')
    # Query either person and vary order/connectives; never always answer the first fact.
    extra_reading=[]
    for r in rows:
        if r['category']=='reading' and r['prompt'].count(' has a ')==2:
            name,color,obj=r['group'].split(':')[1:]
            ni=names.index(name)
            other=names[(ni+1)%len(names)]
            ci=colors.index(color)
            distractor=colors[(ci+2)%6]
            extra_reading += [dict(r,prompt=f"{other} has a {distractor} {obj}. {name} has a {color} {obj}. What color is {name}'s {obj}?"),
                             dict(r,prompt=f"{other} has a {distractor} {obj}, but {name} has a {color} {obj}. What color is {name}'s {obj}?")]
    rows.extend(extra_reading)
    for q in ['hello friend','hi friend','greetings','good day','good morning bro','good evening bro']:
        add('chat','greeting',q,'Hi! How can I help you?','train')
    # Retain familiar chat behavior from the previous checkpoint; exact duplicate
    # messages and all reserved evaluation prompts are excluded by the runner.
    return rows
