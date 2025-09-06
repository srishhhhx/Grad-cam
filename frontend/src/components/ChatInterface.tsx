import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Send,
  Bot,
  User,
  CheckCircle,
  ArrowRight,
  AlertCircle,
  FileText
} from 'lucide-react';
import { UNetPrediction, ChatMessage, PatientProfile, ChatSession } from '../types';
import { startChatbotConversation, sendChatbotMessage, chatbotHealthCheck } from '../utils/api';

interface ChatInterfaceProps {
  prediction: UNetPrediction;
  onComplete: (profile: PatientProfile, session: ChatSession) => void;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({ prediction, onComplete }) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [isComplete, setIsComplete] = useState(false);
  const [showReportButton, setShowReportButton] = useState(false);
  const [conversationId, setConversationId] = useState<string>('');
  const [patientContext, setPatientContext] = useState<any>({});
  const [currentQuestion, setCurrentQuestion] = useState<any>(null);
  const [chatbotAvailable, setChatbotAvailable] = useState<boolean>(true);
  const [error, setError] = useState<string>('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Check chatbot availability and start conversation
    initializeChatbot();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const initializeChatbot = async () => {
    try {
      // Check if chatbot is available
      const healthResponse = await chatbotHealthCheck();
      if (!healthResponse.data?.chatbot_available) {
        setChatbotAvailable(false);
        setError('Chatbot service is not available. Please try again later.');
        return;
      }

      // Start conversation with scan results
      const scanResults = prediction?.prediction || null;
      const response = await startChatbotConversation(scanResults);

      if (response.success && response.data) {
        setConversationId(response.data.conversation_id);
        setPatientContext(response.data.patient_context);
        setCurrentQuestion(response.data.current_question);

        // Add the initial bot message
        addBotMessage(response.data.message);
      } else {
        throw new Error('Failed to start conversation');
      }
    } catch (error) {
      console.error('Error initializing chatbot:', error);
      setChatbotAvailable(false);
      setError('Failed to connect to the chatbot. Please refresh the page and try again.');

      // Fallback to a simple message
      addBotMessage("I'm having trouble connecting to the AI assistant. Please refresh the page and try again, or contact support if the issue persists.");
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const addBotMessage = (text: string) => {
    setIsTyping(true);
    setTimeout(() => {
      const newMessage: ChatMessage = {
        id: Date.now().toString(),
        type: 'assistant',
        content: text,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, newMessage]);
      setIsTyping(false);
    }, 1000 + Math.random() * 1000); // Simulate typing delay
  };

  const addUserMessage = (text: string) => {
    const newMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: text,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, newMessage]);
    return newMessage;
  };

  const handleSendMessage = async () => {
    if (!currentMessage.trim() || !chatbotAvailable || !conversationId) return;

    const userMessage = addUserMessage(currentMessage);
    const messageText = currentMessage;
    setCurrentMessage('');
    setError('');

    try {
      setIsTyping(true);

      // Prepare conversation history for API
      const conversationHistory = [...messages, userMessage].map(msg => ({
        id: msg.id,
        type: msg.type,
        content: msg.content,
        timestamp: msg.timestamp.toISOString()
      }));

      // Send message to chatbot
      const response = await sendChatbotMessage(
        messageText,
        conversationId,
        patientContext,
        conversationHistory,
        prediction?.prediction || null
      );

      setIsTyping(false);

      if (response.success && response.data) {
        // Update state with response
        setPatientContext(response.data.patient_context);
        setCurrentQuestion(response.data.current_question);

        if (response.data.error) {
          setError('Please check your response and try again.');
        }

        // Add bot response
        const botMessage: ChatMessage = {
          id: Date.now().toString(),
          type: 'assistant',
          content: response.data.message,
          timestamp: new Date()
        };
        setMessages(prev => [...prev, botMessage]);

        // Check if questionnaire is complete
        if (response.data.questionnaire_complete && !isComplete) {
          setIsComplete(true);
          // Don't automatically show report button - let user continue chatting
        }
      } else {
        throw new Error('Failed to get response from chatbot');
      }
    } catch (error) {
      console.error('Error sending message:', error);
      setIsTyping(false);
      setError('Sorry, I encountered an issue. Please try again.');

      // Add fallback message
      const fallbackMessage: ChatMessage = {
        id: Date.now().toString(),
        type: 'assistant',
        content: "I apologize, but I'm having trouble processing your response right now. Could you please try again?",
        timestamp: new Date()
      };
      setMessages(prev => [...prev, fallbackMessage]);
    }
  };

  const handleShowReportButton = () => {
    setShowReportButton(true);
  };

  const handleProceedToReport = () => {
    // Create patient profile from the context data
    const completeProfile: PatientProfile = {
      id: Date.now().toString(),
      demographics: patientContext.demographics || {},
      menstrualHistory: patientContext.menstrual_history || {},
      symptoms: patientContext.symptoms || {},
      reproductiveHistory: patientContext.reproductive_history || {},
      medicalHistory: patientContext.medical_history || {},
      lifestyleFactors: patientContext.lifestyle_factors || {},
      createdAt: new Date(),
      updatedAt: new Date()
    } as PatientProfile;

    const chatSession: ChatSession = {
      id: conversationId,
      patientId: 'current-patient',
      messages,
      patientProfile: completeProfile,
      isComplete: true,
      createdAt: new Date()
    };

    onComplete(completeProfile, chatSession);
  };

  const handleQuickResponse = (response: string) => {
    setCurrentMessage(response);
    setTimeout(() => handleSendMessage(), 100);
  };

  // Calculate progress based on patient context
  const calculateProgress = () => {
    if (!patientContext) return 0;

    let completedFields = 0;
    let totalFields = 6; // Approximate number of key questions

    if (patientContext.demographics?.age) completedFields++;
    if (patientContext.demographics?.sex) completedFields++;
    if (patientContext.symptoms?.heavy_bleeding !== undefined) completedFields++;
    if (patientContext.symptoms?.pelvic_pain !== undefined) completedFields++;
    if (patientContext.medical_history?.family_history_fibroids) completedFields++;
    if (patientContext.demographics?.ethnicity || completedFields >= 5) completedFields++; // Optional field

    return Math.min((completedFields / totalFields) * 100, 100);
  };

  const progress = calculateProgress();

  return (
    <div className="max-w-4xl mx-auto">
      {/* Progress Bar */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-medical-700">Health Information Progress</span>
          <span className="text-sm text-medical-600">
            {isComplete ? 'Complete' : `${Math.round(progress)}% Complete`}
          </span>
        </div>
        <div className="progress-bar">
          <motion.div
            className="progress-fill"
            initial={{ width: 0 }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.5 }}
          />
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg flex items-center space-x-2"
        >
          <AlertCircle className="w-4 h-4 text-red-500 flex-shrink-0" />
          <span className="text-sm text-red-700">{error}</span>
        </motion.div>
      )}

      {/* Chatbot Unavailable Warning */}
      {!chatbotAvailable && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-4 p-4 bg-yellow-50 border border-yellow-200 rounded-lg"
        >
          <div className="flex items-center space-x-2 mb-2">
            <AlertCircle className="w-5 h-5 text-yellow-600" />
            <span className="font-medium text-yellow-800">AI Assistant Unavailable</span>
          </div>
          <p className="text-sm text-yellow-700">
            The AI-powered chatbot is currently unavailable. Please refresh the page or contact support if the issue persists.
          </p>
        </motion.div>
      )}

      {/* Chat Container */}
      <div className="chat-container h-96 overflow-y-auto p-6 mb-4">
        <div className="space-y-4">
          <AnimatePresence>
            {messages.map((message) => (
              <motion.div
                key={message.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div className={`flex items-start space-x-3 max-w-xs lg:max-w-md ${message.type === 'user' ? 'flex-row-reverse space-x-reverse' : ''
                  }`}>
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${message.type === 'user'
                    ? 'bg-primary-500'
                    : 'bg-gradient-to-br from-medical-500 to-medical-600'
                    }`}>
                    {message.type === 'user' ? (
                      <User className="w-4 h-4 text-white" />
                    ) : (
                      <Bot className="w-4 h-4 text-white" />
                    )}
                  </div>
                  <div className={`${message.type === 'user' ? 'chat-message-user' : 'chat-message-assistant'
                    }`}>
                    <p className="text-sm">{message.content}</p>
                    <p className="text-xs opacity-70 mt-1">
                      {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </p>
                  </div>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>

          {/* Typing Indicator */}
          <AnimatePresence>
            {isTyping && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="flex justify-start"
              >
                <div className="flex items-start space-x-3">
                  <div className="w-8 h-8 bg-gradient-to-br from-medical-500 to-medical-600 rounded-full flex items-center justify-center">
                    <Bot className="w-4 h-4 text-white" />
                  </div>
                  <div className="chat-message-assistant">
                    <div className="flex space-x-1">
                      <motion.div
                        animate={{ scale: [1, 1.2, 1] }}
                        transition={{ duration: 1, repeat: Infinity, delay: 0 }}
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
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      {!isComplete && (
        <div className="space-y-4">
          {/* Quick Response Options */}
          {currentQuestion?.type === 'choice' && currentQuestion.options && (
            <div className="flex flex-wrap gap-2">
              {currentQuestion.options.map((option: string) => (
                <button
                  key={option}
                  onClick={() => handleQuickResponse(option)}
                  className="btn-secondary text-sm"
                  disabled={isTyping || !chatbotAvailable}
                >
                  {option}
                </button>
              ))}
            </div>
          )}

          {/* Text Input */}
          <div className="flex space-x-3">
            <input
              type={currentQuestion?.type === 'number' ? 'number' : 'text'}
              value={currentMessage}
              onChange={(e) => setCurrentMessage(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && handleSendMessage()}
              placeholder={
                !chatbotAvailable ? 'Chatbot unavailable...' :
                  currentQuestion?.type === 'number' ? 'Enter a number...' :
                    currentQuestion?.type === 'choice' ? 'Type your answer or use buttons above...' :
                      'Type your answer...'
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

          {currentQuestion?.optional && (
            <p className="text-sm text-medical-500 text-center">
              This question is optional. You can skip it by typing "skip".
            </p>
          )}

          {chatbotAvailable && (
            <p className="text-sm text-medical-400 text-center">
              💡 Powered by Google Gemini AI for intelligent health conversations
            </p>
          )}
        </div>
      )}

      {/* Continue Conversation After Completion */}
      {isComplete && !showReportButton && (
        <div className="space-y-4">
          <div className="text-center p-4 bg-green-50 border border-green-200 rounded-lg">
            <p className="text-green-700 text-sm">
              ✅ Health questionnaire completed! You can continue chatting or proceed to your report.
            </p>
          </div>

          {/* Text Input for continued conversation */}
          <div className="flex space-x-3">
            <input
              type="text"
              value={currentMessage}
              onChange={(e) => setCurrentMessage(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && handleSendMessage()}
              placeholder="Ask me anything about your results or health..."
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

          <div className="flex justify-center">
            <button
              onClick={handleShowReportButton}
              className="btn-primary flex items-center space-x-2"
            >
              <FileText className="w-4 h-4" />
              <span>Generate My Report</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>

          {chatbotAvailable && (
            <p className="text-sm text-medical-400 text-center">
              💡 Feel free to ask follow-up questions about your scan results or health recommendations
            </p>
          )}
        </div>
      )}

      {/* Completion Message */}
      {showReportButton && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center medical-glass p-6"
        >
          <div className="w-16 h-16 mx-auto mb-4 bg-green-100 rounded-full flex items-center justify-center">
            <CheckCircle className="w-8 h-8 text-green-600" />
          </div>
          <h3 className="text-lg font-semibold text-medical-800 mb-2">
            Ready to Generate Your Report
          </h3>
          <p className="text-medical-600 mb-6">
            Thank you for providing your health information. Your personalized medical report is ready to be generated.
          </p>
          <button
            onClick={handleProceedToReport}
            className="btn-primary flex items-center space-x-2 mx-auto"
          >
            <FileText className="w-5 h-5" />
            <span>Generate My Medical Report</span>
            <ArrowRight className="w-5 h-5" />
          </button>
        </motion.div>
      )}
    </div>
  );
};

export default ChatInterface;
