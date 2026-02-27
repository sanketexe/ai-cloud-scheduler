import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Avatar,
  Chip,
  IconButton,
  Menu,
  MenuItem,
  Alert,
  Skeleton,
  Tooltip,
  Fab,
  Collapse,
} from '@mui/material';
import {
  Chat as ChatIcon,
  Send as SendIcon,
  Psychology as AIIcon,
  Person as PersonIcon,
  MoreVert as MoreVertIcon,
  Clear as ClearIcon,
  ContentCopy as CopyIcon,
  ThumbUp as ThumbUpIcon,
  ThumbDown as ThumbDownIcon,
  BarChart as ChartIcon,
  TrendingUp as TrendingUpIcon,
  AttachMoney as CostIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
} from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';
import { useMutation, useQuery } from 'react-query';

interface ChatMessage {
  id: string;
  type: 'user' | 'ai';
  content: string;
  timestamp: Date;
  visualizations?: Visualization[];
  recommendations?: Recommendation[];
  followUpQuestions?: string[];
  confidence?: number;
  sources?: string[];
}

interface Visualization {
  type: 'line' | 'bar' | 'pie';
  title: string;
  data: any[];
  description: string;
}

interface Recommendation {
  title: string;
  description: string;
  impact: string;
  priority: 'high' | 'medium' | 'low';
  actionable: boolean;
}

interface ConversationContext {
  conversationId: string;
  userId: string;
  currentFocus: string;
  sessionHistory: ChatMessage[];
}

const COLORS = ['#2196f3', '#4caf50', '#ff9800', '#f44336', '#9c27b0', '#00bcd4'];

const sampleQuestions = [
  "What are my top cost drivers this month?",
  "Show me anomalies in my AWS spending",
  "How can I optimize my EC2 costs?",
  "Compare my costs across different regions",
  "What's the trend in my database spending?",
  "Recommend reserved instances for my workload",
];

const NaturalLanguageChat: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: '1',
      type: 'ai',
      content: "Hello! I'm your AI-powered FinOps assistant. I can help you analyze costs, identify optimization opportunities, and answer questions about your cloud spending. What would you like to know?",
      timestamp: new Date(),
      followUpQuestions: sampleQuestions.slice(0, 3),
      confidence: 1.0,
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [conversationId] = useState(() => `conv_${Date.now()}`);
  const [menuAnchor, setMenuAnchor] = useState<null | HTMLElement>(null);
  const [selectedMessageId, setSelectedMessageId] = useState<string | null>(null);
  const [showSuggestions, setShowSuggestions] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Mutation for sending messages
  const sendMessageMutation = useMutation(
    async (message: string) => {
      const response = await fetch('/api/ai/natural-language/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: message,
          conversationId,
          context: {
            previousMessages: messages.slice(-5), // Send last 5 messages for context
          },
        }),
      });
      if (!response.ok) throw new Error('Failed to send message');
      return response.json();
    },
    {
      onSuccess: (response) => {
        const aiMessage: ChatMessage = {
          id: `ai_${Date.now()}`,
          type: 'ai',
          content: response.answer,
          timestamp: new Date(),
          visualizations: response.visualizations || [],
          recommendations: response.recommendations || [],
          followUpQuestions: response.followUpQuestions || [],
          confidence: response.confidence || 0.8,
          sources: response.sources || [],
        };
        setMessages(prev => [...prev, aiMessage]);
        setIsTyping(false);
      },
      onError: () => {
        const errorMessage: ChatMessage = {
          id: `error_${Date.now()}`,
          type: 'ai',
          content: "I'm sorry, I encountered an error processing your request. Please try again or rephrase your question.",
          timestamp: new Date(),
          confidence: 0,
        };
        setMessages(prev => [...prev, errorMessage]);
        setIsTyping(false);
      },
    }
  );

  const handleSendMessage = (message?: string) => {
    const messageText = message || inputValue.trim();
    if (!messageText) return;

    // Add user message
    const userMessage: ChatMessage = {
      id: `user_${Date.now()}`,
      type: 'user',
      content: messageText,
      timestamp: new Date(),
    };
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsTyping(true);
    setShowSuggestions(false);

    // Send to AI
    sendMessageMutation.mutate(messageText);
  };

  const handleKeyPress = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSendMessage();
    }
  };

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>, messageId: string) => {
    setMenuAnchor(event.currentTarget);
    setSelectedMessageId(messageId);
  };

  const handleMenuClose = () => {
    setMenuAnchor(null);
    setSelectedMessageId(null);
  };

  const handleCopyMessage = () => {
    const message = messages.find(m => m.id === selectedMessageId);
    if (message) {
      navigator.clipboard.writeText(message.content);
    }
    handleMenuClose();
  };

  const handleClearChat = () => {
    setMessages([messages[0]]); // Keep the initial AI greeting
    setShowSuggestions(true);
    handleMenuClose();
  };

  const renderVisualization = (viz: Visualization, index: number) => {
    const chartHeight = 300;
    
    switch (viz.type) {
      case 'line':
        return (
          <Box key={index} sx={{ mb: 3 }}>
            <Typography variant="subtitle2" gutterBottom>
              {viz.title}
            </Typography>
            <Box sx={{ height: chartHeight }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={viz.data}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <RechartsTooltip />
                  <Line type="monotone" dataKey="value" stroke="#2196f3" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </Box>
            <Typography variant="caption" color="text.secondary">
              {viz.description}
            </Typography>
          </Box>
        );
      
      case 'bar':
        return (
          <Box key={index} sx={{ mb: 3 }}>
            <Typography variant="subtitle2" gutterBottom>
              {viz.title}
            </Typography>
            <Box sx={{ height: chartHeight }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={viz.data}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <RechartsTooltip />
                  <Bar dataKey="value" fill="#2196f3" />
                </BarChart>
              </ResponsiveContainer>
            </Box>
            <Typography variant="caption" color="text.secondary">
              {viz.description}
            </Typography>
          </Box>
        );
      
      case 'pie':
        return (
          <Box key={index} sx={{ mb: 3 }}>
            <Typography variant="subtitle2" gutterBottom>
              {viz.title}
            </Typography>
            <Box sx={{ height: chartHeight }}>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={viz.data}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {viz.data.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <RechartsTooltip />
                </PieChart>
              </ResponsiveContainer>
            </Box>
            <Typography variant="caption" color="text.secondary">
              {viz.description}
            </Typography>
          </Box>
        );
      
      default:
        return null;
    }
  };

  const renderRecommendations = (recommendations: Recommendation[]) => (
    <Box sx={{ mt: 2 }}>
      <Typography variant="subtitle2" gutterBottom>
        Recommendations
      </Typography>
      {recommendations.map((rec, index) => (
        <Alert
          key={index}
          severity={rec.priority === 'high' ? 'error' : rec.priority === 'medium' ? 'warning' : 'info'}
          sx={{ mb: 1 }}
          action={
            rec.actionable && (
              <Button size="small" color="inherit">
                Apply
              </Button>
            )
          }
        >
          <Typography variant="subtitle2">{rec.title}</Typography>
          <Typography variant="body2">{rec.description}</Typography>
          <Typography variant="caption" color="text.secondary">
            Expected Impact: {rec.impact}
          </Typography>
        </Alert>
      ))}
    </Box>
  );

  return (
    <Box sx={{ p: 3, height: '600px', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
        <ChatIcon sx={{ color: 'primary.main' }} />
        <Typography variant="h6">Natural Language Assistant</Typography>
        <Box sx={{ ml: 'auto' }}>
          <Tooltip title="Clear Chat">
            <IconButton onClick={handleClearChat} size="small">
              <ClearIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* Chat Messages */}
      <Box sx={{ flexGrow: 1, overflow: 'auto', mb: 2, pr: 1 }}>
        {messages.map((message) => (
          <Box
            key={message.id}
            sx={{
              display: 'flex',
              mb: 3,
              flexDirection: message.type === 'user' ? 'row-reverse' : 'row',
            }}
          >
            <Avatar
              sx={{
                bgcolor: message.type === 'user' ? 'primary.main' : 'secondary.main',
                mx: 1,
              }}
            >
              {message.type === 'user' ? <PersonIcon /> : <AIIcon />}
            </Avatar>
            
            <Card
              sx={{
                maxWidth: '70%',
                bgcolor: message.type === 'user' ? 'primary.main' : 'background.paper',
                color: message.type === 'user' ? 'primary.contrastText' : 'text.primary',
              }}
            >
              <CardContent sx={{ pb: '16px !important' }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }}>
                    {message.content}
                  </Typography>
                  <IconButton
                    size="small"
                    onClick={(e) => handleMenuOpen(e, message.id)}
                    sx={{ ml: 1, opacity: 0.7 }}
                  >
                    <MoreVertIcon fontSize="small" />
                  </IconButton>
                </Box>

                {/* Visualizations */}
                {message.visualizations && message.visualizations.length > 0 && (
                  <Box sx={{ mt: 2 }}>
                    {message.visualizations.map((viz, index) => renderVisualization(viz, index))}
                  </Box>
                )}

                {/* Recommendations */}
                {message.recommendations && message.recommendations.length > 0 && 
                  renderRecommendations(message.recommendations)
                }

                {/* Follow-up Questions */}
                {message.followUpQuestions && message.followUpQuestions.length > 0 && (
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="caption" color="text.secondary" gutterBottom>
                      Suggested follow-up questions:
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 1 }}>
                      {message.followUpQuestions.map((question, index) => (
                        <Chip
                          key={index}
                          label={question}
                          size="small"
                          onClick={() => handleSendMessage(question)}
                          sx={{ cursor: 'pointer' }}
                        />
                      ))}
                    </Box>
                  </Box>
                )}

                {/* Confidence and Sources */}
                {message.type === 'ai' && (
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mt: 2 }}>
                    {message.confidence !== undefined && (
                      <Typography variant="caption" color="text.secondary">
                        Confidence: {(message.confidence * 100).toFixed(0)}%
                      </Typography>
                    )}
                    <Box>
                      <Tooltip title="Helpful">
                        <IconButton size="small">
                          <ThumbUpIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Not Helpful">
                        <IconButton size="small">
                          <ThumbDownIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    </Box>
                  </Box>
                )}
              </CardContent>
            </Card>
          </Box>
        ))}

        {/* Typing Indicator */}
        {isTyping && (
          <Box sx={{ display: 'flex', mb: 3 }}>
            <Avatar sx={{ bgcolor: 'secondary.main', mx: 1 }}>
              <AIIcon />
            </Avatar>
            <Card sx={{ maxWidth: '70%' }}>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Typography variant="body2" color="text.secondary">
                    AI is thinking...
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 0.5 }}>
                    <Box
                      sx={{
                        width: 6,
                        height: 6,
                        borderRadius: '50%',
                        bgcolor: 'primary.main',
                        animation: 'pulse 1.5s ease-in-out infinite',
                      }}
                    />
                    <Box
                      sx={{
                        width: 6,
                        height: 6,
                        borderRadius: '50%',
                        bgcolor: 'primary.main',
                        animation: 'pulse 1.5s ease-in-out infinite 0.2s',
                      }}
                    />
                    <Box
                      sx={{
                        width: 6,
                        height: 6,
                        borderRadius: '50%',
                        bgcolor: 'primary.main',
                        animation: 'pulse 1.5s ease-in-out infinite 0.4s',
                      }}
                    />
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Box>
        )}

        <div ref={messagesEndRef} />
      </Box>

      {/* Suggested Questions */}
      <Collapse in={showSuggestions}>
        <Box sx={{ mb: 2 }}>
          <Typography variant="caption" color="text.secondary" gutterBottom>
            Try asking:
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 1 }}>
            {sampleQuestions.map((question, index) => (
              <Chip
                key={index}
                label={question}
                size="small"
                variant="outlined"
                onClick={() => handleSendMessage(question)}
                sx={{ cursor: 'pointer' }}
              />
            ))}
          </Box>
        </Box>
      </Collapse>

      {/* Input Area */}
      <Box sx={{ display: 'flex', gap: 1, alignItems: 'flex-end' }}>
        <TextField
          ref={inputRef}
          fullWidth
          multiline
          maxRows={4}
          placeholder="Ask me anything about your cloud costs..."
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyPress={handleKeyPress}
          disabled={isTyping}
          variant="outlined"
          size="small"
        />
        <Button
          variant="contained"
          onClick={() => handleSendMessage()}
          disabled={!inputValue.trim() || isTyping}
          sx={{ minWidth: 'auto', px: 2 }}
        >
          <SendIcon />
        </Button>
      </Box>

      {/* Context Menu */}
      <Menu
        anchorEl={menuAnchor}
        open={Boolean(menuAnchor)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={handleCopyMessage}>
          <CopyIcon sx={{ mr: 1 }} fontSize="small" />
          Copy Message
        </MenuItem>
      </Menu>

      {/* Pulse Animation Keyframes */}
      <style>
        {`
          @keyframes pulse {
            0%, 70%, 100% {
              opacity: 0.4;
              transform: scale(1);
            }
            35% {
              opacity: 1;
              transform: scale(1.2);
            }
          }
        `}
      </style>
    </Box>
  );
};

export default NaturalLanguageChat;