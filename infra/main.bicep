// Azure Container Apps + Azure AI Search + Azure OpenAI for the RAG agent.
// Deploy: az deployment group create -g <rg> -f infra/main.bicep -p infra/main.parameters.json
// NOTE: written for this project but never deployed (no Azure subscription was used).

targetScope = 'resourceGroup'

@description('Azure region. Pick one where your Azure OpenAI models are available.')
param location string = resourceGroup().location

@description('Short prefix for resource names (lowercase letters and digits).')
@minLength(3)
@maxLength(12)
param namePrefix string = 'ragagent'

@description('Container image. The quickstart image is a placeholder until you push your own.')
param containerImage string = 'mcr.microsoft.com/k8se/quickstart:latest'

param chatModelName string = 'gpt-4o-mini'
param chatModelVersion string = '2024-07-18'
param embeddingModelName string = 'text-embedding-3-small'
param embeddingModelVersion string = '1'

@allowed(['free', 'basic', 'standard'])
param searchSku string = 'basic'

var suffix = uniqueString(resourceGroup().id)
var openAiName = '${namePrefix}-aoai-${suffix}'
var searchName = '${namePrefix}-search-${suffix}'
var acrName = '${namePrefix}acr${suffix}'

resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: '${namePrefix}-logs-${suffix}'
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: acrName
  location: location
  sku: { name: 'Basic' }
  properties: { adminUserEnabled: true } // simple for a demo; prefer managed identity + AcrPull
}

resource openAi 'Microsoft.CognitiveServices/accounts@2024-10-01' = {
  name: openAiName
  location: location
  kind: 'OpenAI'
  sku: { name: 'S0' }
  properties: {
    customSubDomainName: openAiName
    publicNetworkAccess: 'Enabled'
  }
}

resource chatDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = {
  parent: openAi
  name: chatModelName
  sku: { name: 'GlobalStandard', capacity: 10 }
  properties: {
    model: { format: 'OpenAI', name: chatModelName, version: chatModelVersion }
  }
}

resource embeddingDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = {
  parent: openAi
  name: embeddingModelName
  sku: { name: 'Standard', capacity: 10 }
  properties: {
    model: { format: 'OpenAI', name: embeddingModelName, version: embeddingModelVersion }
  }
  dependsOn: [chatDeployment] // deployments on one account must be created one at a time
}

resource search 'Microsoft.Search/searchServices@2023-11-01' = {
  name: searchName
  location: location
  sku: { name: searchSku }
  properties: {
    replicaCount: 1
    partitionCount: 1
  }
}

resource env 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: '${namePrefix}-env-${suffix}'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logs.properties.customerId
        sharedKey: logs.listKeys().primarySharedKey
      }
    }
  }
}

resource app 'Microsoft.App/containerApps@2024-03-01' = {
  name: '${namePrefix}-app'
  location: location
  properties: {
    managedEnvironmentId: env.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8000
      }
      registries: [
        {
          server: acr.properties.loginServer
          username: acr.listCredentials().username
          passwordSecretRef: 'acr-password'
        }
      ]
      secrets: [
        { name: 'acr-password', value: acr.listCredentials().passwords[0].value }
        { name: 'openai-key', value: openAi.listKeys().key1 }
        { name: 'search-key', value: search.listAdminKeys().primaryKey }
      ]
    }
    template: {
      containers: [
        {
          name: 'api'
          image: containerImage
          resources: { cpu: json('0.5'), memory: '1Gi' }
          env: [
            { name: 'RAG_BACKEND', value: 'azure' }
            { name: 'RAG_MIN_SCORE', value: '0.0' }
            { name: 'AZURE_OPENAI_ENDPOINT', value: openAi.properties.endpoint }
            { name: 'AZURE_OPENAI_API_KEY', secretRef: 'openai-key' }
            { name: 'AZURE_OPENAI_CHAT_DEPLOYMENT', value: chatDeployment.name }
            { name: 'AZURE_OPENAI_EMBEDDING_DEPLOYMENT', value: embeddingDeployment.name }
            { name: 'AZURE_SEARCH_ENDPOINT', value: 'https://${search.name}.search.windows.net' }
            { name: 'AZURE_SEARCH_API_KEY', secretRef: 'search-key' }
            { name: 'AZURE_SEARCH_INDEX', value: 'hr-policies' }
          ]
          probes: [
            {
              type: 'Liveness'
              httpGet: { path: '/health', port: 8000 }
            }
          ]
        }
      ]
      scale: { minReplicas: 0, maxReplicas: 2 }
    }
  }
}

output appUrl string = 'https://${app.properties.configuration.ingress.fqdn}'
output containerAppName string = app.name
output acrName string = acr.name
output acrLoginServer string = acr.properties.loginServer
