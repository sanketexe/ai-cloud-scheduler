import React, { useState, useRef, useEffect } from 'react';
import {
    Box,
    Paper,
    Typography,
    TextField,
    IconButton,
    Avatar,
    Fab,
    Zoom,
    Fade,
    List,
    ListItem,
    ListItemText,
    CircularProgress,
    Chip,
    Tooltip,
} from '@mui/material';
import {
    Chat as ChatIcon,
    Close as CloseIcon,
    Send as SendIcon,
    Psychology as AIIcon,
    SmartToy as RobotIcon,
    DeleteOutline as ClearIcon,
} from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';
import { useLocation } from 'react-router-dom';

interface Message {
    role: 'user' | 'assistant';
    content: string;
    timestamp: string;
}

const PlatformFloatingChat: React.FC = () => {
    const [isOpen, setIsOpen] = useState(false);
    const [input, setInput] = useState('');
    const [messages, setMessages] = useState<Message[]>([
        {
            role: 'assistant',
            content: "Hi there! I'm CloudPilot, your AI platform assistant. Ask me anything about your costs, migration, or just 'How many instances am I running?'",
            timestamp: new Date().toISOString(),
        },
    ]);
    const [isLoading, setIsLoading] = useState(false);
    const [suggestions, setSuggestions] = useState<string[]>([
        "How many instances are running?",
        "How can I save money?",
        "Tell me about migration planning",
    ]);

    const messagesEndRef = useRef<HTMLDivElement>(null);
    const location = useLocation();

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        if (isOpen) {
            scrollToBottom();
        }
    }, [messages, isOpen]);

    const handleSend = async (text: string) => {
        const messageText = text || input.trim();
        if (!messageText || isLoading) return;

        const userMsg: Message = {
            role: 'user',
            content: messageText,
            timestamp: new Date().toISOString(),
        };

        setMessages((prev) => [...prev, userMsg]);
        setInput('');
        setIsLoading(true);

        try {
            const response = await fetch('/api/v1/ai/assistant/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: messageText,
                    context: {
                        page: location.pathname,
                    },
                    conversation_history: messages.map(m => ({
                        role: m.role,
                        content: m.content
                    }))
                }),
            });

            const data = await response.json();

            const assistantMsg: Message = {
                role: 'assistant',
                content: data.message,
                timestamp: data.timestamp || new Date().toISOString(),
            };

            setMessages((prev) => [...prev, assistantMsg]);
            if (data.suggestions) setSuggestions(data.suggestions);

        } catch (error) {
            console.error('Chat error:', error);
            setMessages((prev) => [
                ...prev,
                {
                    role: 'assistant',
                    content: "I'm having trouble connecting to my brain right now. Please check if the backend is running!",
                    timestamp: new Date().toISOString(),
                },
            ]);
        } finally {
            setIsLoading(false);
        }
    };

    const clearChat = () => {
        setMessages([messages[0]]);
    };

    return (
        <>
            {/* Floating Action Button */}
            <Box sx={{ position: 'fixed', bottom: 24, right: 24, zIndex: 1000 }}>
                <Zoom in={true}>
                    <Fab
                        color="primary"
                        aria-label="chat"
                        onClick={() => setIsOpen(!isOpen)}
                        sx={{
                            width: 60,
                            height: 60,
                            boxShadow: '0 8px 32px rgba(79, 70, 229, 0.3)',
                            background: 'linear-gradient(135deg, #6366f1 0%, #4f46e5 100%)',
                            '&:hover': {
                                transform: 'scale(1.1) rotate(5deg)',
                                transition: 'all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275)',
                                boxShadow: '0 12px 40px rgba(79, 70, 229, 0.4)',
                            },
                        }}
                    >
                        {isOpen ? <CloseIcon /> : <ChatIcon />}
                    </Fab>
                </Zoom>
            </Box>

            {/* Chat Window */}
            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        initial={{ opacity: 0, y: 100, scale: 0.8, x: 50 }}
                        animate={{ opacity: 1, y: 0, scale: 1, x: 0 }}
                        exit={{ opacity: 0, y: 100, scale: 0.8, x: 50 }}
                        transition={{ type: 'spring', damping: 25, stiffness: 200 }}
                        style={{
                            position: 'fixed',
                            bottom: 100,
                            right: 24,
                            width: 380,
                            height: 550,
                            zIndex: 1000,
                            pointerEvents: 'auto',
                        }}
                    >
                        <Paper
                            elevation={24}
                            sx={{
                                height: '100%',
                                display: 'flex',
                                flexDirection: 'column',
                                borderRadius: 4,
                                overflow: 'hidden',
                                background: 'rgba(255, 255, 255, 0.95)',
                                backdropFilter: 'blur(10px)',
                                border: '1px solid rgba(0,0,0,0.1)',
                            }}
                        >
                            {/* Header */}
                            <Box
                                sx={{
                                    p: 2,
                                    background: 'linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)',
                                    color: 'white',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'space-between',
                                    boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
                                }}
                            >
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                                    <Avatar sx={{ bgcolor: 'rgba(255,255,255,0.2)', color: 'white', backdropFilter: 'blur(4px)' }}>
                                        <RobotIcon />
                                    </Avatar>
                                    <Box>
                                        <Typography variant="subtitle1" fontWeight="bold">
                                            CloudPilot AI
                                        </Typography>
                                        <Typography variant="caption" sx={{ opacity: 0.8 }}>
                                            Online | Real-time Data Active
                                        </Typography>
                                    </Box>
                                </Box>
                                <IconButton size="small" onClick={clearChat} sx={{ color: 'white' }}>
                                    <ClearIcon fontSize="small" />
                                </IconButton>
                            </Box>

                            {/* Chat Area */}
                            <Box sx={{ flex: 1, p: 2, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 2 }}>
                                {messages.map((msg, i) => (
                                    <Box
                                        key={i}
                                        sx={{
                                            alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start',
                                            maxWidth: '85%',
                                        }}
                                    >
                                        <Paper
                                            elevation={0}
                                            sx={{
                                                p: 1.5,
                                                borderRadius: 3,
                                                bgcolor: msg.role === 'user' ? '#4f46e5' : '#f8fafd',
                                                color: msg.role === 'user' ? 'white' : '#1e293b',
                                                borderBottomRightRadius: msg.role === 'user' ? 2 : 12,
                                                borderBottomLeftRadius: msg.role === 'assistant' ? 2 : 12,
                                                boxShadow: msg.role === 'assistant' ? '0 2px 8px rgba(0,0,0,0.05)' : '0 4px 12px rgba(79, 70, 229, 0.2)',
                                            }}
                                        >
                                            <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                                                {msg.content}
                                            </Typography>
                                        </Paper>
                                        <Typography
                                            variant="caption"
                                            color="text.secondary"
                                            sx={{ mt: 0.5, display: 'block', textAlign: msg.role === 'user' ? 'right' : 'left' }}
                                        >
                                            {new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                        </Typography>
                                    </Box>
                                ))}
                                {isLoading && (
                                    <Box sx={{ alignSelf: 'flex-start', ml: 1 }}>
                                        <CircularProgress size={20} thickness={5} />
                                    </Box>
                                )}
                                <div ref={messagesEndRef} />
                            </Box>

                            {/* Suggestions */}
                            {!isLoading && suggestions.length > 0 && (
                                <Box sx={{ px: 2, pb: 1, display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                                    {suggestions.map((s, i) => (
                                        <Chip
                                            key={i}
                                            label={s}
                                            size="small"
                                            onClick={() => handleSend(s)}
                                            sx={{
                                                fontSize: '0.7rem',
                                                cursor: 'pointer',
                                                bgcolor: 'rgba(79, 70, 229, 0.08)',
                                                color: '#4f46e5',
                                                border: '1px solid rgba(79, 70, 229, 0.2)',
                                                '&:hover': { bgcolor: 'rgba(79, 70, 229, 0.15)', transform: 'translateY(-1px)' },
                                                transition: 'all 0.2s ease',
                                            }}
                                        />
                                    ))}
                                </Box>
                            )}

                            {/* Input Area */}
                            <Box sx={{ p: 2, borderTop: '1px solid rgba(0,0,0,0.05)' }}>
                                <Box sx={{ display: 'flex', gap: 1 }}>
                                    <TextField
                                        fullWidth
                                        size="small"
                                        placeholder="Ask CloudPilot..."
                                        value={input}
                                        onChange={(e) => setInput(e.target.value)}
                                        onKeyPress={(e) => e.key === 'Enter' && handleSend('')}
                                        sx={{
                                            '& .MuiOutlinedInput-root': {
                                                borderRadius: 3,
                                                bgcolor: '#f8f9fa',
                                            },
                                        }}
                                    />
                                    <IconButton
                                        disabled={!input.trim() || isLoading}
                                        onClick={() => handleSend('')}
                                        sx={{
                                            bgcolor: '#4f46e5',
                                            color: 'white',
                                            '&:hover': { bgcolor: '#4338ca', transform: 'scale(1.05)' },
                                            '&.Mui-disabled': { bgcolor: '#e2e8f0', color: '#94a3b8' },
                                            transition: 'all 0.2s ease',
                                        }}
                                    >
                                        <SendIcon fontSize="small" />
                                    </IconButton>
                                </Box>
                            </Box>
                        </Paper>
                    </motion.div>
                )}
            </AnimatePresence>
        </>
    );
};

export default PlatformFloatingChat;
