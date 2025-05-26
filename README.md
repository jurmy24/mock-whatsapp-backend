This is a simple backend for Twiga for AI development.

How to run:
Fill in the `.env`
* Put the Neon DB URI (that I share with you) into DATABASE_URL
* Put your own Together AI API key into LLM_API_KEY (you usually get 5 dollars in free credits)

Run the commands:
`poetry install`
or 
`uv install`

And then:
`source .venv/bin/activate` if on mac/linux, `.venv\Scripts\activate` if on windows. 

Then write:
`uvicorn app.main:app --port 8000 --reload` to start the development FastAPI server

You'll see that the LLM calls aren't implemented yet though.
