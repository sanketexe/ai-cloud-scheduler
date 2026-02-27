#!/usr/bin/env python3
"""
LLM Provider Abstraction for Migration Chatbot
Supports Gemini, Hugging Face, and other providers
"""

import os
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    async def generate_response(self, prompt: str, context: Dict = None) -> str:
        """Generate response from the LLM"""
        pass
    
    @abstractmethod
    def get_embeddings(self, text: str) -> List[float]:
        """Get text embeddings for vector search"""
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the provider name"""
        pass


class GeminiProvider(LLMProvider):
    """Google Gemini API provider"""
    
    def __init__(self):
        try:
            import google.generativeai as genai
            self.genai = genai
            
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable is required")
            
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            
            # Initialize embeddings
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            logger.info("Gemini provider initialized successfully")
            
        except ImportError:
            raise ImportError("Please install google-generativeai: pip install google-generativeai")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini provider: {e}")
            raise
    
    async def generate_response(self, prompt: str, context: Dict = None) -> str:
        """Generate response using Gemini API"""
        try:
            # Enhance prompt with context
            enhanced_prompt = self._enhance_prompt_with_context(prompt, context)
            
            # Generate response
            response = await asyncio.to_thread(
                self.model.generate_content, enhanced_prompt
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return f"I apologize, but I encountered an error processing your request. Please try again or rephrase your question."
    
    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings using sentence-transformers"""
        try:
            embeddings = self.embedding_model.encode(text)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return []
    
    def get_provider_name(self) -> str:
        return "gemini"
    
    def _enhance_prompt_with_context(self, prompt: str, context: Dict = None) -> str:
        """Enhance prompt with user context and migration expertise"""
        
        system_prompt = """You are an expert cloud migration consultant specializing in helping startups and businesses migrate to the cloud. 

Your expertise includes:
- AWS, GCP, and Azure platform comparison and migration strategies
- Cost optimization and budget planning for cloud migrations
- Security, compliance, and regulatory considerations
- Migration timeline planning and risk assessment
- Startup-specific challenges and resource constraints
- Technical architecture recommendations

Guidelines for responses:
1. Provide practical, actionable advice tailored to the user's specific situation
2. Always consider cost implications and budget constraints
3. Mention specific cloud services and tools when relevant
4. Include potential risks and mitigation strategies
5. Suggest next steps or follow-up actions
6. Keep responses concise but comprehensive (aim for 200-400 words)
7. Use a friendly, professional tone suitable for startup founders and technical teams

"""
        
        # Add user context if available
        if context:
            context_info = "\n\nUser Context:\n"
            if context.get('company_size'):
                context_info += f"- Company Size: {context['company_size']}\n"
            if context.get('industry'):
                context_info += f"- Industry: {context['industry']}\n"
            if context.get('current_infrastructure'):
                context_info += f"- Current Infrastructure: {context['current_infrastructure']}\n"
            if context.get('budget_range'):
                context_info += f"- Budget Range: ${context['budget_range']}/month\n"
            
            system_prompt += context_info
        
        enhanced_prompt = f"{system_prompt}\n\nUser Question: {prompt}\n\nResponse:"
        
        return enhanced_prompt


class HuggingFaceProvider(LLMProvider):
    """Hugging Face API provider"""
    
    def __init__(self):
        try:
            from transformers import pipeline
            import torch
            
            # Check if CUDA is available
            device = 0 if torch.cuda.is_available() else -1
            
            # Initialize text generation pipeline
            # Using a conversational model suitable for chat
            self.generator = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-medium",
                tokenizer="microsoft/DialoGPT-medium",
                device=device,
                max_length=512,
                do_sample=True,
                temperature=0.7,
                pad_token_id=50256
            )
            
            # Initialize embeddings
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            logger.info("Hugging Face provider initialized successfully")
            
        except ImportError:
            raise ImportError("Please install transformers and torch: pip install transformers torch")
        except Exception as e:
            logger.error(f"Failed to initialize Hugging Face provider: {e}")
            raise
    
    async def generate_response(self, prompt: str, context: Dict = None) -> str:
        """Generate response using Hugging Face model"""
        try:
            # Enhance prompt with context
            enhanced_prompt = self._enhance_prompt_with_context(prompt, context)
            
            # Generate response
            response = await asyncio.to_thread(
                self._generate_text, enhanced_prompt
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Hugging Face API error: {e}")
            return f"I apologize, but I encountered an error processing your request. Please try again or rephrase your question."
    
    def _generate_text(self, prompt: str) -> str:
        """Generate text using the pipeline"""
        try:
            # Generate response
            outputs = self.generator(
                prompt,
                max_new_tokens=200,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.generator.tokenizer.eos_token_id
            )
            
            # Extract generated text
            generated_text = outputs[0]['generated_text']
            
            # Remove the input prompt from the response
            if prompt in generated_text:
                response = generated_text.replace(prompt, "").strip()
            else:
                response = generated_text.strip()
            
            # Clean up the response
            response = self._clean_response(response)
            
            return response if response else "I'd be happy to help with your cloud migration questions. Could you please provide more specific details about your requirements?"
            
        except Exception as e:
            logger.error(f"Text generation error: {e}")
            return "I encountered an issue generating a response. Please try rephrasing your question."
    
    def _clean_response(self, response: str) -> str:
        """Clean and format the generated response"""
        # Remove common artifacts
        response = response.replace("<|endoftext|>", "")
        response = response.replace("</s>", "")
        response = response.strip()
        
        # Ensure response is not too short or repetitive
        if len(response) < 20:
            return ""
        
        return response
    
    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings using sentence-transformers"""
        try:
            embeddings = self.embedding_model.encode(text)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return []
    
    def get_provider_name(self) -> str:
        return "huggingface"
    
    def _enhance_prompt_with_context(self, prompt: str, context: Dict = None) -> str:
        """Enhance prompt with context for better responses"""
        
        # Create a conversational prompt
        system_context = "You are a helpful cloud migration consultant. "
        
        if context:
            if context.get('company_size'):
                system_context += f"The user works at a {context['company_size']} company. "
            if context.get('industry'):
                system_context += f"They are in the {context['industry']} industry. "
        
        enhanced_prompt = f"{system_context}User asks: {prompt}\nAssistant:"
        
        return enhanced_prompt


class OllamaProvider(LLMProvider):
    """Ollama local model provider (optional)"""
    
    def __init__(self):
        try:
            import requests
            self.base_url = "http://localhost:11434"
            self.model_name = "llama2"  # or any other model you have installed
            
            # Test connection
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                raise ConnectionError("Cannot connect to Ollama server")
            
            # Initialize embeddings
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            logger.info("Ollama provider initialized successfully")
            
        except ImportError:
            raise ImportError("Please install requests: pip install requests")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama provider: {e}")
            raise
    
    async def generate_response(self, prompt: str, context: Dict = None) -> str:
        """Generate response using Ollama API"""
        try:
            import requests
            
            enhanced_prompt = self._enhance_prompt_with_context(prompt, context)
            
            response = await asyncio.to_thread(
                requests.post,
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": enhanced_prompt,
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                raise Exception(f"Ollama API error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            return f"I apologize, but I encountered an error processing your request. Please try again."
    
    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings using sentence-transformers"""
        try:
            embeddings = self.embedding_model.encode(text)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return []
    
    def get_provider_name(self) -> str:
        return "ollama"
    
    def _enhance_prompt_with_context(self, prompt: str, context: Dict = None) -> str:
        """Enhance prompt with context"""
        system_prompt = """You are an expert cloud migration consultant. Provide helpful, practical advice for cloud migration questions. Keep responses concise and actionable.

"""
        
        if context:
            system_prompt += f"User context: {context}\n\n"
        
        return f"{system_prompt}Question: {prompt}\n\nAnswer:"


def get_llm_provider() -> LLMProvider:
    """Factory function to get the configured LLM provider"""
    
    provider_type = os.getenv("LLM_PROVIDER", "gemini").lower()
    
    try:
        if provider_type == "gemini":
            return GeminiProvider()
        elif provider_type == "huggingface":
            return HuggingFaceProvider()
        elif provider_type == "ollama":
            return OllamaProvider()
        else:
            logger.warning(f"Unknown LLM provider: {provider_type}, falling back to Gemini")
            return GeminiProvider()
            
    except Exception as e:
        logger.error(f"Failed to initialize {provider_type} provider: {e}")
        
        # Fallback to a simple mock provider for development
        logger.info("Falling back to mock provider for development")
        return MockProvider()


class MockProvider(LLMProvider):
    """Mock provider for development/testing"""
    
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Mock provider initialized (for development)")
    
    async def generate_response(self, prompt: str, context: Dict = None) -> str:
        """Generate mock response"""
        
        # Simple keyword-based responses for testing
        prompt_lower = prompt.lower()
        
        if "cost" in prompt_lower:
            return """Based on your requirements, here's a cost comparison:

**AWS**: Estimated $2,500-3,500/month
- EC2 instances: $1,200/month
- RDS database: $800/month
- S3 storage: $300/month
- Additional services: $700/month

**GCP**: Estimated $2,200-3,200/month
- Compute Engine: $1,000/month
- Cloud SQL: $700/month
- Cloud Storage: $250/month
- Additional services: $750/month

**Azure**: Estimated $2,400-3,400/month
- Virtual Machines: $1,100/month
- Azure SQL: $750/month
- Blob Storage: $280/month
- Additional services: $720/month

**Recommendation**: GCP offers the best value for your startup, with excellent Kubernetes support and competitive pricing."""
        
        elif "timeline" in prompt_lower or "time" in prompt_lower:
            return """Here's a typical migration timeline for your project:

**Phase 1: Assessment & Planning (2-3 weeks)**
- Infrastructure discovery and dependency mapping
- Migration strategy design
- Team training and preparation

**Phase 2: Environment Setup (1-2 weeks)**
- Cloud account setup and configuration
- Network connectivity and security setup
- CI/CD pipeline configuration

**Phase 3: Migration Execution (4-6 weeks)**
- Database migration with minimal downtime
- Application migration and testing
- DNS cutover and traffic routing

**Phase 4: Optimization (2-3 weeks)**
- Performance tuning and monitoring setup
- Cost optimization and right-sizing
- Documentation and knowledge transfer

**Total Estimated Timeline**: 9-14 weeks

This can be accelerated with proper planning and dedicated resources."""
        
        elif "security" in prompt_lower or "compliance" in prompt_lower:
            return """Key security considerations for your cloud migration:

**Data Protection**:
- Enable encryption at rest and in transit
- Implement proper key management
- Set up data backup and disaster recovery

**Access Control**:
- Use multi-factor authentication (MFA)
- Implement role-based access control (RBAC)
- Regular access reviews and audits

**Network Security**:
- Configure VPCs and security groups
- Set up network monitoring and logging
- Implement DDoS protection

**Compliance Requirements**:
- GDPR compliance for EU data
- SOC 2 Type II certification
- Industry-specific requirements (HIPAA, PCI-DSS)

**Monitoring & Incident Response**:
- Set up security monitoring and alerting
- Develop incident response procedures
- Regular security assessments and penetration testing

All major cloud providers (AWS, GCP, Azure) offer comprehensive security tools and compliance certifications."""
        
        else:
            return f"""Thank you for your question about cloud migration. 

Based on your inquiry, I'd recommend starting with a comprehensive assessment of your current infrastructure and business requirements. 

Key considerations include:
- Current application architecture and dependencies
- Performance and scalability requirements  
- Budget constraints and cost optimization goals
- Security and compliance requirements
- Team expertise and training needs

Would you like me to dive deeper into any specific aspect of your migration planning? I can help with cost comparisons, timeline estimation, or technical architecture recommendations.

Feel free to ask about specific cloud providers (AWS, GCP, Azure) or particular aspects of your migration journey."""
    
    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings using sentence-transformers"""
        try:
            embeddings = self.embedding_model.encode(text)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return []
    
    def get_provider_name(self) -> str:
        return "mock"


# Test function
async def test_providers():
    """Test all available providers"""
    
    test_prompt = "I'm a startup looking to migrate to the cloud. Which provider should I choose?"
    test_context = {
        "company_size": "STARTUP",
        "industry": "Technology",
        "budget_range": "5000"
    }
    
    providers = ["gemini", "huggingface", "mock"]
    
    for provider_name in providers:
        try:
            os.environ["LLM_PROVIDER"] = provider_name
            provider = get_llm_provider()
            
            print(f"\n=== Testing {provider.get_provider_name()} Provider ===")
            response = await provider.generate_response(test_prompt, test_context)
            print(f"Response: {response[:200]}...")
            
        except Exception as e:
            print(f"Error testing {provider_name}: {e}")


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_providers())