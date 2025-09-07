import os
from crewai import LLM
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import (
	SerplyWebSearchTool,
	SerplyScholarSearchTool
)
from surgical_safety_monitoring_crew.tools.surgical_object_detection_api import SurgicalObjectDetectionAPI



@CrewBase
class SurgicalSafetyMonitoringCrewCrew:
    """SurgicalSafetyMonitoringCrew crew"""

    
    @agent
    def video_frame_intake_specialist(self) -> Agent:
        
        return Agent(
            config=self.agents_config["video_frame_intake_specialist"],
            tools=[
				SurgicalObjectDetectionAPI()
            ],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            max_execution_time=None,
            llm=LLM(
                model="gemini/gemini-1.5-flash",
                temperature=0.7,
            ),
        )
    
    @agent
    def ppe_detection_specialist(self) -> Agent:
        
        return Agent(
            config=self.agents_config["ppe_detection_specialist"],
            tools=[
				SurgicalObjectDetectionAPI()
            ],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            max_execution_time=None,
            llm=LLM(
                model="gemini/gemini-2.5-flash",
                temperature=0.7,
            ),
        )
    
    @agent
    def surgical_compliance_rules_auditor(self) -> Agent:
        
        return Agent(
            config=self.agents_config["surgical_compliance_rules_auditor"],
            tools=[
				SerplyWebSearchTool(),
				SerplyScholarSearchTool()
            ],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            max_execution_time=None,
            llm=LLM(
                model="gemini/gemini-2.5-flash",
                temperature=0.7,
            ),
        )
    
    @agent
    def privacy_compliance_specialist(self) -> Agent:
        
        return Agent(
            config=self.agents_config["privacy_compliance_specialist"],
            tools=[
				SurgicalObjectDetectionAPI()
            ],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            max_execution_time=None,
            llm=LLM(
                model="gemini/gemini-2.5-flash",
                temperature=0.7,
            ),
        )
    
    @agent
    def or_safety_report_compiler(self) -> Agent:
        
        return Agent(
            config=self.agents_config["or_safety_report_compiler"],
            tools=[

            ],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            max_execution_time=None,
            llm=LLM(
                model="gemini/gemini-2.5-flash",
                temperature=0.7,
            ),
        )
    

    
    @task
    def process_video_frames_at_standard_rate(self) -> Task:
        return Task(
            config=self.tasks_config["process_video_frames_at_standard_rate"],
            markdown=False,
        )
    
    @task
    def detect_ppe_and_surgical_equipment(self) -> Task:
        return Task(
            config=self.tasks_config["detect_ppe_and_surgical_equipment"],
            markdown=False,
        )
    
    @task
    def audit_surgical_safety_compliance(self) -> Task:
        return Task(
            config=self.tasks_config["audit_surgical_safety_compliance"],
            markdown=False,
        )
    
    @task
    def anonymize_patient_data(self) -> Task:
        return Task(
            config=self.tasks_config["anonymize_patient_data"],
            markdown=False,
        )
    
    @task
    def compile_safety_report(self) -> Task:
        return Task(
            config=self.tasks_config["compile_safety_report"],
            markdown=False,
            output_file= 'safety_report/OR_safety_report.md'
        )
    

    @crew
    def crew(self) -> Crew:
        """Creates the SurgicalSafetyMonitoringCrew crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
