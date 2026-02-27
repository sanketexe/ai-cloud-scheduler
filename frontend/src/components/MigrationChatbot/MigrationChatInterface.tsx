import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  TextField,
  Button,
  Paper,
  Typography,
  Chip,
  Avatar,
  IconButton,
  Collapse,
  Card,
  CardContent,
  Divider,
  CircularProgress,
  Tooltip,
  Badge
} from '@mui/material';
import {
  Send,
  SmartToy,
  Person,
  ExpandMore,
  ExpandLess,
  Lightbulb,
  TrendingUp,
  Security,
  Schedule,
  AttachMoney,
  CloudUpload
} from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';
import toast from 'react-hot-toast';

interface ChatMessage {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  metadata?: {
    sources?: string[];
    suggestedActions?: string[];
    followUpQuestions?: string[];
    confidence?: number;
  };
}

interface MigrationChatInterfaceProps {
  onAssessmentStart?: () => void;
  userContext?: {
    company_size?: string;
    industry?: string;
    current_infrastructure?: string;
    budget_range?: string;
  };
}

const MigrationChatInterface: React.FC<MigrationChatInterfaceProps> = ({
  onAssessmentStart,
  userContext
}) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversationId, setConversationId] = useState<string>('');
  const [expandedSources, setExpandedSources] = useState<string>('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    initializeChat();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const initializeChat = async () => {
    try {
      // Get conversation starters
      const response = await fetch('/api/migration/suggestions');
      const data = await response.json();

      const welcomeMessage: ChatMessage = {
        id: '1',
        type: 'assistant',
        content: `👋 Hi! I'm your **AI Migration Assistant**. I'm here to help you navigate your cloud migration journey.

I can help you with:
🔍 **Cloud Provider Selection** - Compare AWS, GCP, and Azure
💰 **Cost Planning** - Estimate migration and operational costs  
📋 **Migration Strategy** - Plan your migration approach
⚡ **Risk Assessment** - Identify and mitigate potential issues
🛡️ **Compliance** - Ensure regulatory requirements are met

**Quick Start Options:**`,
        timestamp: new Date(),
        metadata: {
          suggestedActions: data.starters.slice(0, 4),
          followUpQuestions: [
            "What's your current infrastructure setup?",
            "What's your primary goal for cloud migration?",
            "Do you have any specific compliance requirements?"
          ]
        }
      };

      setMessages([welcomeMessage]);
    } catch (error) {
      console.error('Failed to initialize chat:', error);
      toast.error('Failed to initialize chat assistant');
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const sendMessage = async (content: string) => {
    if (!content.trim()) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      content,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await fetch('/api/migration/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: content,
          user_context: userContext,
          conversation_id: conversationId
        })
      });

      if (!response.ok) {
        throw new Error('Failed to get response');
      }

      const data = await response.json();

      const assistantMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: data.response,
        timestamp: new Date(),
        metadata: {
          sources: data.sources,
          suggestedActions: data.suggested_actions,
          followUpQuestions: data.follow_up_questions,
          confidence: data.confidence
        }
      };

      setMessages(prev => [...prev, assistantMessage]);
      setConversationId(data.conversation_id);

    } catch (error) {
      console.error('Error sending message:', error);
      toast.error('Failed to send message. Please try again.');
      
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: 'I apologize, but I encountered an error. Please try rephrasing your question or contact support if the issue persists.',
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSuggestedAction = (action: string) => {
    if (action.includes('assessment') || action.includes('wizard')) {
      onAssessmentStart?.();
    } else {
      sendMessage(action);
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'success';
    if (confidence >= 0.6) return 'warning';
    return 'error';
  };

  const getConfidenceLabel = (confidence: number) => {
    if (confidence >= 0.8) return 'High Confidence';
    if (confidence >= 0.6) return 'Medium Confidence';
    return 'Low Confidence';
  };

  return (
    <Box sx={{ height: '600px', display: 'flex', flexDirection: 'column', bgcolor: 'background.paper', borderRadius: 2 }}>
      {/* Header */}
      <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider', bgcolor: 'primary.main', color: 'white', borderRadius: '8px 8px 0 0' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Avatar sx={{ bgcolor: 'white', color: 'primary.main' }}>
            <SmartToy />
          </Avatar>
          <Box>
            <Typography variant="h6">Migration Assistant</Typography>
            <Typography variant="body2" sx={{ opacity: 0.9 }}>
              AI-powered cloud migration advisor
            </Typography>
          </Box>
        </Box>
      </Box>

      {/* Messages Area */}
      <Box sx={{ flex: 1, overflow: 'auto', p: 2, bgcolor: 'grey.50' }}>
        <AnimatePresence>
          {messages.map((message) => (
            <MessageBubble
              key={message.id}
              message={message}
              onSuggestedAction={handleSuggestedAction}
              expandedSources={expandedSources}
              onToggleSources={setExpandedSources}
            />
          ))}
        </AnimatePresence>
        
        {isLoading && <TypingIndicator />}
        <div ref={messagesEndRef} />
      </Box>

      {/* Input Area */}
      <Box sx={{ p: 2, borderTop: 1, borderColor: 'divider', bgcolor: 'background.paper' }}>
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'flex-end' }}>
          <TextField
            fullWidth
            multiline
            maxRows={3}
            variant="outlined"
            placeholder="Ask about cloud migration, costs, timelines, or best practices..."
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage(inputValue);
              }
            }}
            disabled={isLoading}
            sx={{ 
              '& .MuiOutlinedInput-root': {
                borderRadius: 3
              }
            }}
          />
          <Button
            variant="contained"
            onClick={() => sendMessage(inputValue)}
            disabled={isLoading || !inputValue.trim()}
            sx={{ 
              minWidth: 'auto', 
              px: 2, 
              py: 1.5,
              borderRadius: 3,
              height: 'fit-content'
            }}
          >
            <Send />
          </Button>
        </Box>
        
        {/* Quick Actions */}
        <Box sx={{ mt: 1, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
          <Chip
            icon={<AttachMoney />}
            label="Compare Costs"
            size="small"
            onClick={() => sendMessage("Compare costs across AWS, GCP, and Azure for my startup")}
            sx={{ cursor: 'pointer' }}
          />
          <Chip
            icon={<Schedule />}
            label="Migration Timeline"
            size="small"
            onClick={() => sendMessage("How long will cloud migration take for my company?")}
            sx={{ cursor: 'pointer' }}
          />
          <Chip
            icon={<Security />}
            label="Security & Compliance"
            size="small"
            onClick={() => sendMessage("What security considerations should I have for cloud migration?")}
            sx={{ cursor: 'pointer' }}
          />
        </Box>
      </Box>
    </Box>
  );
};

const MessageBubble: React.FC<{
  message: ChatMessage;
  onSuggestedAction: (action: string) => void;
  expandedSources: string;
  onToggleSources: (messageId: string) => void;
}> = ({ message, onSuggestedAction, expandedSources, onToggleSources }) => {
  const isUser = message.type === 'user';
  const isExpanded = expandedSources === message.id;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.3 }}
    >
      <Box sx={{ mb: 3, display: 'flex', justifyContent: isUser ? 'flex-end' : 'flex-start' }}>
        <Box sx={{ maxWidth: '80%', display: 'flex', gap: 1, alignItems: 'flex-start' }}>
          {!isUser && (
            <Avatar sx={{ bgcolor: 'primary.main', width: 32, height: 32 }}>
              <SmartToy sx={{ fontSize: 18 }} />
            </Avatar>
          )}
          
          <Box>
            <Paper
              sx={{
                p: 2,
                bgcolor: isUser ? 'primary.main' : 'white',
                color: isUser ? 'white' : 'text.primary',
                borderRadius: isUser ? '20px 20px 4px 20px' : '20px 20px 20px 4px',
                boxShadow: 2
              }}
            >
              <Typography 
                variant="body1" 
                sx={{ 
                  whiteSpace: 'pre-wrap',
                  '& strong': { fontWeight: 600 },
                  '& em': { fontStyle: 'italic' }
                }}
              >
                {message.content}
              </Typography>

              {/* Confidence Score */}
              {message.metadata?.confidence !== undefined && (
                <Box sx={{ mt: 1, display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Tooltip title={`AI Confidence: ${(message.metadata.confidence * 100).toFixed(0)}%`}>
                    <Badge
                      badgeContent={`${(message.metadata.confidence * 100).toFixed(0)}%`}
                      color={getConfidenceColor(message.metadata.confidence)}
                      sx={{ fontSize: '0.7rem' }}
                    >
                      <Chip
                        size="small"
                        label={getConfidenceLabel(message.metadata.confidence)}
                        color={getConfidenceColor(message.metadata.confidence)}
                        variant="outlined"
                      />
                    </Badge>
                  </Tooltip>
                </Box>
              )}
            </Paper>

            {/* Suggested Actions */}
            {message.metadata?.suggestedActions && message.metadata.suggestedActions.length > 0 && (
              <Box sx={{ mt: 2, display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                {message.metadata.suggestedActions.map((action, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: index * 0.1 }}
                  >
                    <Chip
                      label={action}
                      onClick={() => onSuggestedAction(action)}
                      size="small"
                      variant="outlined"
                      sx={{ 
                        cursor: 'pointer',
                        '&:hover': { bgcolor: 'primary.light', color: 'white' }
                      }}
                      icon={<Lightbulb sx={{ fontSize: 16 }} />}
                    />
                  </motion.div>
                ))}
              </Box>
            )}

            {/* Follow-up Questions */}
            {message.metadata?.followUpQuestions && message.metadata.followUpQuestions.length > 0 && (
              <Card sx={{ mt: 2, bgcolor: 'grey.50' }}>
                <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    💡 You might also ask:
                  </Typography>
                  {message.metadata.followUpQuestions.map((question, index) => (
                    <Typography
                      key={index}
                      variant="body2"
                      sx={{
                        cursor: 'pointer',
                        display: 'block',
                        mt: 0.5,
                        p: 0.5,
                        borderRadius: 1,
                        '&:hover': { bgcolor: 'primary.light', color: 'white' }
                      }}
                      onClick={() => onSuggestedAction(question)}
                    >
                      • {question}
                    </Typography>
                  ))}
                </CardContent>
              </Card>
            )}

            {/* Sources */}
            {message.metadata?.sources && message.metadata.sources.length > 0 && (
              <Box sx={{ mt: 1 }}>
                <Button
                  size="small"
                  onClick={() => onToggleSources(isExpanded ? '' : message.id)}
                  endIcon={isExpanded ? <ExpandLess /> : <ExpandMore />}
                  sx={{ textTransform: 'none', fontSize: '0.75rem' }}
                >
                  Sources ({message.metadata.sources.length})
                </Button>
                <Collapse in={isExpanded}>
                  <Box sx={{ mt: 1, p: 1, bgcolor: 'grey.100', borderRadius: 1 }}>
                    {message.metadata.sources.map((source, index) => (
                      <Typography key={index} variant="caption" display="block">
                        • {source}
                      </Typography>
                    ))}
                  </Box>
                </Collapse>
              </Box>
            )}
          </Box>

          {isUser && (
            <Avatar sx={{ bgcolor: 'grey.400', width: 32, height: 32 }}>
              <Person sx={{ fontSize: 18 }} />
            </Avatar>
          )}
        </Box>
      </Box>
    </motion.div>
  );
};

const TypingIndicator: React.FC = () => (
  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
    <Avatar sx={{ bgcolor: 'primary.main', width: 32, height: 32 }}>
      <SmartToy sx={{ fontSize: 18 }} />
    </Avatar>
    <Paper sx={{ p: 2, borderRadius: '20px 20px 20px 4px' }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <CircularProgress size={16} />
        <Typography variant="body2" color="text.secondary">
          Thinking...
        </Typography>
      </Box>
    </Paper>
  </Box>
);

// Helper function to get confidence color
const getConfidenceColor = (confidence: number): 'success' | 'warning' | 'error' => {
  if (confidence >= 0.8) return 'success';
  if (confidence >= 0.6) return 'warning';
  return 'error';
};

// Helper function to get confidence label
const getConfidenceLabel = (confidence: number): string => {
  if (confidence >= 0.8) return 'High Confidence';
  if (confidence >= 0.6) return 'Medium Confidence';
  return 'Low Confidence';
};

export default MigrationChatInterface;