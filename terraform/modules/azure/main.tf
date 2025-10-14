# Azure Infrastructure Module for Cloud Intelligence Platform

variable "environment" {
  type = string
}

variable "project_name" {
  type = string
}

variable "region" {
  type = string
}

variable "common_tags" {
  type = map(string)
}

# Resource Group
resource "azurerm_resource_group" "main" {
  name     = "${var.project_name}-${var.environment}-rg"
  location = var.region
  tags     = var.common_tags
}

# AKS Cluster
resource "azurerm_kubernetes_cluster" "main" {
  name                = "${var.project_name}-${var.environment}-aks"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  dns_prefix          = "${var.project_name}-${var.environment}"

  default_node_pool {
    name       = "default"
    node_count = 3
    vm_size    = "Standard_D2_v2"
  }

  identity {
    type = "SystemAssigned"
  }

  tags = var.common_tags
}

# Outputs
output "cluster_endpoint" {
  value = azurerm_kubernetes_cluster.main.kube_config.0.host
}

output "cluster_name" {
  value = azurerm_kubernetes_cluster.main.name
}