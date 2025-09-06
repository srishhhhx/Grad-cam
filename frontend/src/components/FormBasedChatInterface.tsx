import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  CheckCircle,
  Bot,
  User,
  Send
} from 'lucide-react';
import { UNetPrediction, ChatMessage, PatientProfile, ChatSession } from '../types';
import { startFormBasedChatbot, sendChatbotMessage, chatbotHealthCheck } from '../utils/api';
import QuestionnaireForm, { QuestionnaireData } from './QuestionnaireForm';

interface FormBasedChatInterfaceProps {
  prediction: UNetPrediction;
  onComplete: (profile: PatientProfile, session: ChatSession) => void;
}

const FormBasedChatInterface: React.FC<FormBasedChatInterfaceProps> = ({
  prediction,
  onComplete
}) => {
  const [currentStep, setCurrentStep] = useState<'form' | 'chat'>('form');
  const [isLoading, setIsLoading] = useState(false);
  const [chatbotAvailable, setChatbotAvailable] = useState(false);

  // Chat state
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [conversationId, setConversationId] = useState('');
  const [patientContext, setPatientContext] = useState<any>(null);

  const [scanResults, setScanResults] = useState<any>(null);

  useEffect(() => {
    checkChatbotHealth();
  }, []);

  const checkChatbotHealth = async () => {
    try {
      const response = await chatbotHealthCheck();
      setChatbotAvailable(response.data?.chatbot_available || false);
    } catch (error) {
      console.error('Chatbot health check failed:', error);
      setChatbotAvailable(false);
    }
  };

  const handleFormSubmit = async (formData: QuestionnaireData) => {
    if (!chatbotAvailable) {
      alert('Chatbot service is not available. Please try again later.');
      return;
    }

    setIsLoading(true);
    try {
      // Prepare scan results
      const scanResultsData = {
        fibroidDetected: prediction.prediction.fibroidDetected,
        fibroidCount: prediction.prediction.fibroidCount,
        confidence: prediction.prediction.confidence,
        severity: prediction.prediction.fibroidAreas?.[0]?.severity || 'mild'
      };

      // Start form-based chatbot conversation
      const response = await startFormBasedChatbot(formData, scanResultsData);

      // Initialize chat with the response
      const botMessage: ChatMessage = {
        id: `msg_${Date.now()}`,
        type: 'assistant',
        content: response.data?.message || 'Hello! I\'m here to help you understand your results.',
        timestamp: new Date()
      };

      setMessages([botMessage]);
      setConversationId(response.data?.conversation_id || '');
      setPatientContext(response.data?.patient_context || {});
      setScanResults(scanResultsData);
      setCurrentStep('chat');
    } catch (error) {
      console.error('Error starting form-based chat:', error);
      alert('Failed to start conversation. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSendMessage = async () => {
    if (!currentMessage.trim() || isTyping || !chatbotAvailable) return;

    const userMessage: ChatMessage = {
      id: `msg_${Date.now()}`,
      type: 'user',
      content: currentMessage.trim(),
      timestamp: new Date()
    };

    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setCurrentMessage('');
    setIsTyping(true);

    try {
      const response = await sendChatbotMessage(
        userMessage.content,
        conversationId,
        patientContext,
        newMessages,
        scanResults
      );

      const botMessage: ChatMessage = {
        id: `msg_${Date.now() + 1}`,
        type: 'assistant',
        content: response.data?.message || 'I understand. Please continue.',
        timestamp: new Date()
      };

      setMessages([...newMessages, botMessage]);
      setPatientContext(response.data?.patient_context || {});
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: ChatMessage = {
        id: `msg_${Date.now() + 1}`,
        type: 'assistant',
        content: 'I apologize, but I encountered an issue processing your message. Could you please try again?',
        timestamp: new Date()
      };
      setMessages([...newMessages, errorMessage]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleGenerateReport = () => {
    // Create patient profile from context
    const profile: PatientProfile = {
      id: Date.now().toString(),
      demographics: patientContext?.demographics || {},
      menstrualHistory: patientContext?.menstrual_history || {},
      symptoms: patientContext?.symptoms || {},
      reproductiveHistory: patientContext?.reproductive_history || {},
      medicalHistory: patientContext?.medical_history || {},
      lifestyleFactors: patientContext?.lifestyle_factors || {},
      riskFactors: [],
      createdAt: new Date(),
      updatedAt: new Date()
    } as PatientProfile;

    // Create chat session
    const session: ChatSession = {
      id: conversationId,
      patientId: 'current-patient',
      messages,
      patientProfile: profile,
      isComplete: true,
      startedAt: new Date(),
      createdAt: new Date()
    };

    onComplete(profile, session);
  };

  if (currentStep === 'form') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-6">
        <QuestionnaireForm
          prediction={prediction}
          onSubmit={handleFormSubmit}
          isLoading={isLoading}
        />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-6">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h2 className="text-3xl font-bold text-white mb-4">
            Health Education Chat
          </h2>
          <p className="text-gray-300 text-lg">
            Ask me anything about your results or uterine fibroids in general.
          </p>
        </div>

        {/* Chat Interface */}
        <div className="glass-card p-6 h-[600px] flex flex-col">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto space-y-4 mb-6">
            {messages.map((message) => (
              <motion.div
                key={message.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[80%] p-4 rounded-lg ${message.type === 'user'
                    ? 'bg-medical-500 text-white'
                    : 'bg-white/90 text-black backdrop-blur-sm'
                    }`}
                >
                  <div className="flex items-start space-x-3">
                    {message.type === 'assistant' && (
                      <Bot className="w-5 h-5 text-medical-400 mt-1 flex-shrink-0" />
                    )}
                    {message.type === 'user' && (
                      <User className="w-5 h-5 text-white mt-1 flex-shrink-0" />
                    )}
                    <div className="flex-1">
                      <p className="whitespace-pre-wrap">{message.content}</p>
                      <p className="text-xs opacity-70 mt-2">
                        {message.timestamp.toLocaleTimeString()}
                      </p>
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}

            {/* Typing indicator */}
            <AnimatePresence>
              {isTyping && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="flex justify-start"
                >
                  <div className="bg-white/90 backdrop-blur-sm p-4 rounded-lg">
                    <div className="flex items-center space-x-3">
                      <Bot className="w-5 h-5 text-medical-400" />
                      <div className="flex space-x-1">
                        <motion.div
                          animate={{ scale: [1, 1.2, 1] }}
                          transition={{ duration: 1, repeat: Infinity }}
                          className="w-2 h-2 bg-medical-400 rounded-full"
                        />
                        <motion.div
                          animate={{ scale: [1, 1.2, 1] }}
                          transition={{ duration: 1, repeat: Infinity, delay: 0.2 }}
                          className="w-2 h-2 bg-medical-400 rounded-full"
                        />
                        <motion.div
                          animate={{ scale: [1, 1.2, 1] }}
                          transition={{ duration: 1, repeat: Infinity, delay: 0.4 }}
                          className="w-2 h-2 bg-medical-400 rounded-full"
                        />
                      </div>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Input Area */}
          <div className="space-y-4">
            <div className="flex space-x-3">
              <input
                type="text"
                value={currentMessage}
                onChange={(e) => setCurrentMessage(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && handleSendMessage()}
                placeholder={
                  !chatbotAvailable ? 'Chatbot unavailable...' :
                    'Ask me anything about your results or health...'
                }
                className="input-glass flex-1"
                disabled={isTyping || !chatbotAvailable}
              />
              <button
                onClick={handleSendMessage}
                disabled={!currentMessage.trim() || isTyping || !chatbotAvailable}
                className="btn-primary px-4"
              >
                <Send className="w-4 h-4" />
              </button>
            </div>

            {/* Generate Report Button */}
            <div className="flex justify-center">
              <button
                onClick={handleGenerateReport}
                className="btn-primary"
              >
                <CheckCircle className="w-4 h-4 mr-2" />
                Generate Report
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FormBasedChatInterface;
