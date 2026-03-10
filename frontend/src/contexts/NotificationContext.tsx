import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import axios from 'axios';
import { useAuth } from './AuthContext';

export interface Notification {
  id: string;
  title: string;
  message: string;
  type: 'info' | 'success' | 'warning' | 'error';
  read: boolean;
  createdAt: string;
  actionUrl?: string;
  actionText?: string;
}

interface NotificationContextType {
  notifications: Notification[];
  unreadCount: number;
  loading: boolean;
  markAsRead: (id: string) => Promise<void>;
  markAllAsRead: () => Promise<void>;
  deleteNotification: (id: string) => Promise<void>;
  refreshNotifications: () => Promise<void>;
}

const NotificationContext = createContext<NotificationContextType | undefined>(undefined);

export const useNotifications = () => {
  const context = useContext(NotificationContext);
  if (context === undefined) {
    throw new Error('useNotifications must be used within a NotificationProvider');
  }
  return context;
};

interface NotificationProviderProps {
  children: ReactNode;
}

export const NotificationProvider: React.FC<NotificationProviderProps> = ({ children }) => {
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [loading, setLoading] = useState(false);
  const { isAuthenticated } = useAuth();

  const unreadCount = notifications.filter(n => !n.read).length;

  // Fetch notifications
  const refreshNotifications = async () => {
    if (!isAuthenticated) return;
    
    setLoading(true);
    try {
      const response = await axios.get('/api/v1/notifications');
      setNotifications(response.data);
    } catch (error) {
      console.error('Failed to fetch notifications:', error);
      // Fallback to demo notifications if API fails
      setNotifications([
        {
          id: '1',
          title: 'Welcome to FinOps Platform',
          message: 'Your account has been successfully created. Start by connecting your AWS account.',
          type: 'success',
          read: false,
          createdAt: new Date().toISOString(),
          actionUrl: '/onboarding',
          actionText: 'Get Started'
        },
        {
          id: '2',
          title: 'Cost Alert',
          message: 'Your AWS spending is 15% higher than last month. Review your resources.',
          type: 'warning',
          read: false,
          createdAt: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
          actionUrl: '/cost-analysis',
          actionText: 'View Details'
        },
        {
          id: '3',
          title: 'Migration Analysis Complete',
          message: 'Your cloud migration assessment is ready for review.',
          type: 'info',
          read: true,
          createdAt: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
          actionUrl: '/migration-wizard',
          actionText: 'View Results'
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  // Mark notification as read
  const markAsRead = async (id: string) => {
    try {
      await axios.patch(`/api/v1/notifications/${id}/read`);
      setNotifications(prev => 
        prev.map(n => n.id === id ? { ...n, read: true } : n)
      );
    } catch (error) {
      console.error('Failed to mark notification as read:', error);
      // Optimistic update even if API fails
      setNotifications(prev => 
        prev.map(n => n.id === id ? { ...n, read: true } : n)
      );
    }
  };

  // Mark all notifications as read
  const markAllAsRead = async () => {
    try {
      await axios.patch('/api/v1/notifications/read-all');
      setNotifications(prev => 
        prev.map(n => ({ ...n, read: true }))
      );
    } catch (error) {
      console.error('Failed to mark all notifications as read:', error);
      // Optimistic update even if API fails
      setNotifications(prev => 
        prev.map(n => ({ ...n, read: true }))
      );
    }
  };

  // Delete notification
  const deleteNotification = async (id: string) => {
    try {
      await axios.delete(`/api/v1/notifications/${id}`);
      setNotifications(prev => prev.filter(n => n.id !== id));
    } catch (error) {
      console.error('Failed to delete notification:', error);
      // Optimistic update even if API fails
      setNotifications(prev => prev.filter(n => n.id !== id));
    }
  };

  // Fetch notifications on mount and when auth changes
  useEffect(() => {
    if (isAuthenticated) {
      refreshNotifications();
      
      // Set up polling for new notifications every 30 seconds
      const interval = setInterval(refreshNotifications, 30000);
      return () => clearInterval(interval);
    } else {
      setNotifications([]);
    }
  }, [isAuthenticated]);

  const value: NotificationContextType = {
    notifications,
    unreadCount,
    loading,
    markAsRead,
    markAllAsRead,
    deleteNotification,
    refreshNotifications,
  };

  return (
    <NotificationContext.Provider value={value}>
      {children}
    </NotificationContext.Provider>
  );
};