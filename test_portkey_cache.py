from app.gateway.client import get_langchain_llm

llm = get_langchain_llm(feature="responder")

prompt = "Explain Kubernetes in exactly three sentences."

print("REQUEST 1")
print(llm.invoke(prompt).content)

print("\nREQUEST 2")
print(llm.invoke(prompt).content)