# Colang intent definitions + flows for the production guardrail system.
# Structure mirrors notebooks/01_guardrails.ipynb Experiment 5:
# off-topic + jailbreak rails stacked with dialog rails (greeting/farewell/capabilities).


COLANG_CONTENT = """
define user ask off topic
  "tell me a joke"
  "what is the capital of france"
  "what is 2 plus 2"
  "what should I eat for dinner"
  "who won the game yesterday"
  "recommend a movie"
  "what is the weather today"
  "tell me about world history"

define bot refuse off topic
  "I'm an Enterprise IT Assistant focused on Kubernetes, Intel hardware, and networking. I can't help with that — ask me a technical question from the knowledge base."

define flow handle off topic
  user ask off topic
  bot refuse off topic


define user ask technical
  "how do I deploy a kubernetes pod"
  "how does a kubernetes service work"
  "what is intel sriov"
  "how do bgp routes work"
  "how do I configure a cronjob"
  "what is an intel nic"
  "how do I troubleshoot networking"

define bot allow technical
  "__RAG_TECHNICAL_QUERY_ALLOWED__"

define flow allow technical query
  user ask technical
  bot allow technical


define user express greeting
  "hello"
  "hi"
  "hey"
  "good morning"
  "what's up"

define bot express greeting
  "Hello! I'm your Enterprise IT Assistant. I specialise in Kubernetes, Intel hardware, and enterprise networking. What can I help you with today?"

define flow greeting
  user express greeting
  bot express greeting


define user ask capabilities
  "what can you do"
  "what do you know"
  "help"
  "what are your capabilities"

define bot explain capabilities
  "I'm an Enterprise AI Assistant with deep expertise in Kubernetes, Intel hardware, and enterprise networking. Ask me anything in these areas!"

define flow capabilities
  user ask capabilities
  bot explain capabilities


define user express farewell
  "bye"
  "goodbye"
  "see you later"

define bot express farewell
  "Goodbye! Feel free to return whenever you have more enterprise IT questions. Have a great day!"

define flow farewell
  user express farewell
  bot express farewell
"""

YAML_CONTENT = """
models:
  - type: main
    engine: openai
    model: gpt-3.5-turbo

instructions:
  - type: general
    content: |
      You are an Enterprise IT Assistant specialising in:
      - Kubernetes (deployment, scaling, operators, networking)
      - Intel hardware (CPUs, FPGAs, NICs, SRIOV)
      - Enterprise networking (SDN, VLANs, BGP, routing)
      Only answer questions about these topics. Be professional and concise.
"""

# Simple, explicit guardrail intents.
RAIL_INDICATORS = [
    "can't help with that — ask me a technical question from the knowledge base",
    "Hello! I'm your Enterprise IT Assistant",
    "Goodbye! Feel free to return whenever you have more enterprise IT questions",
    "I'm an Enterprise AI Assistant with deep expertise in Kubernetes, Intel hardware, and enterprise networking",
]

TECHNICAL_QUERY_ALLOWED = "__RAG_TECHNICAL_QUERY_ALLOWED__"

OFF_TOPIC_RESPONSE = (
    "I'm an Enterprise IT Assistant focused on Kubernetes, Intel hardware, "
    "and networking. I can't help with that — ask me a technical question "
    "from the knowledge base."
)

GREETING_RESPONSE = (
    "Hello! I'm your Enterprise IT Assistant. I specialise in Kubernetes, Intel "
    "hardware, and enterprise networking. What can I help you with today?"
)

CAPABILITIES_RESPONSE = (
    "I'm an Enterprise AI Assistant with deep expertise in Kubernetes, Intel "
    "hardware, and enterprise networking. Ask me anything in these areas!"
)

FAREWELL_RESPONSE = (
    "Goodbye! Feel free to return whenever you have more enterprise IT questions. "
    "Have a great day!"
)

KNOWN_RAIL_RESPONSES = [
    OFF_TOPIC_RESPONSE,
    GREETING_RESPONSE,
    FAREWELL_RESPONSE,
    CAPABILITIES_RESPONSE,
]

TECHNICAL_KEYWORDS = [
    "kubernetes",
    "pod",
    "deployment",
    "service",
    "cluster",
    "cronjob",
    "intel",
    "cpu",
    "fpga",
    "nic",
    "sriov",
    "network",
    "networking",
    "vlan",
    "bgp",
    "routing",
    "sdn",
    "operator",
    "autoscale",
    "autoscaling",
]

OFF_TOPIC_PATTERNS = [
    "joke",
    "capital of france",
    "what is 2 plus 2",
    "eat for dinner",
    "movie",
    "weather today",
    "world history",
    "history",
    "recipe",
    "math",
    "poem",
]

MEMORY_QUESTION_PATTERNS = [
    "what is my name",
    "who am i",
    "what did i say my name was",
    "what did i tell you my name was",
    "what did i ask earlier",
    "what did i say earlier",
    "what did i ask before",
    "who are you talking to",
    "what did i tell you",
    "what is your name",
    "what is my username",
]

MEMORY_NAME_PATTERNS = [
    "my name is",
    "i am ",
    "i'm ",
    "call me ",
]

GREETING_KEYWORDS = ["hello", "hi", "hey", "good morning", "what's up"]
CAPABILITY_KEYWORDS = ["what can you do", "what do you know", "help", "what are your capabilities"]
FAREWELL_KEYWORDS = ["bye", "goodbye", "see you later"]
