import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException # type: ignore
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm=ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
    # groq_api_key=os.getenv("GROQ_API_KEY")
    groq_api_key=os.getenv("GROQ_API_KEY")
        )
     
     
   # Function for collecting job posting from the urls 
    def extract_post(self,clean_text):
        prompt_extract=PromptTemplate.from_template(
            """
            ###SCRAPED TEXT FROM WEBSITE
            {page_data}
            ###INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing
            following keys:'role','experience','skills' and 'description'.
            Only return valid JSON.
            ###VALID JSON (NO PREAMBLE )
            """
        )
        chain_extract=prompt_extract | self.llm
        res=chain_extract.invoke(input={'page_data':clean_text})
        try:
            json_parser=JsonOutputParser()
            json_response=json_parser.parse(res.content)
            json_response=json_response[0]
        except OutputParserException:
            raise OutputParserException("Too large for parsing")
        return json_response if isinstance(json_response,list) else [json_response]
    
        
        
    def email_writing(self,job_des,links):
        prompt_email=PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}
            
            ### INSTRUCTION:
            You are Harshdeep, a business development executive at STechs. STechs is an AI & Software Consulting company dedicated to facilitating
            the seamless integration of business processes through automated tools. 
            Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability, 
            process optimization, cost reduction, and heightened overall efficiency. 
            Your job is to write a cold email to the client company's appropriate authorities regarding the job mentioned above describing the capability of STechs 
            in fulfilling their needs.
            Also add the most relevant ones from the following links to showcase STechs's portfolio: {link_list}
            Remember you are Harshdeep, BDE at STechs. 
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):
            
            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job_des), "link_list": links}) 
        return res.content

