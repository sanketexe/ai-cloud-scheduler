# Main Terraform configuration for Cloud Intelligence Platform
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
  }
}

# Variables
variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "cloud-intelligence"
}

variable "region" {
  description = "Primary deployment region"
  type        = string
  default     = "us-west-2"
}

variable "enable_aws" {
  description = "Enable AWS deployment"
  type        = bool
  default     = true
}

variable "enable_gcp" {
  description = "Enable GCP deployment"
  type        = bool
  default     = false
}

variable "enable_azure" {
  description = "Enable Azure deployment"
  type        = bool
  default     = false
}

# Local values
locals {
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
  
  name_prefix = "${var.project_name}-${var.environment}"
}

# AWS Provider Configuration
provider "aws" {
  count  = var.enable_aws ? 1 : 0
  region = var.region
  
  default_tags {
    tags = local.common_tags
  }
}

# GCP Provider Configuration
provider "google" {
  count   = var.enable_gcp ? 1 : 0
  project = var.project_name
  region  = var.region
}

# Azure Provider Configuration
provider "azurerm" {
  count = var.enable_azure ? 1 : 0
  features {}
}

# Conditional module calls
module "aws_infrastructure" {
  count  = var.enable_aws ? 1 : 0
  source = "./modules/aws"
  
  environment  = var.environment
  project_name = var.project_name
  region       = var.region
  common_tags  = local.common_tags
}

module "gcp_infrastructure" {
  count  = var.enable_gcp ? 1 : 0
  source = "./modules/gcp"
  
  environment  = var.environment
  project_name = var.project_name
  region       = var.region
}

module "azure_infrastructure" {
  count  = var.enable_azure ? 1 : 0
  source = "./modules/azure"
  
  environment  = var.environment
  project_name = var.project_name
  region       = var.region
  common_tags  = local.common_tags
}

# Outputs
output "aws_cluster_endpoint" {
  value = var.enable_aws ? module.aws_infrastructure[0].cluster_endpoint : null
}

output "gcp_cluster_endpoint" {
  value = var.enable_gcp ? module.gcp_infrastructure[0].cluster_endpoint : null
}

output "azure_cluster_endpoint" {
  value = var.enable_azure ? module.azure_infrastructure[0].cluster_endpoint : null
}