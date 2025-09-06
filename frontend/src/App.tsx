import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Stethoscope, Users, Activity, Shield, AlertTriangle } from 'lucide-react';
import DoctorInterface from './components/DoctorInterface';
import PatientInterface from './components/PatientInterface';
import TroubleshootingGuide from './components/TroubleshootingGuide';
import { User } from './types';
import { healthCheck } from './utils/api';

function App() {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [showTroubleshooting, setShowTroubleshooting] = useState(false);

  useEffect(() => {
    // Check API health on app load
    checkApiHealth();

    // Simulate loading time for better UX
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 2000);

    return () => clearTimeout(timer);
  }, []);

  const checkApiHealth = async () => {
    try {
      await healthCheck();
      setApiStatus('online');
      console.log('✅ Backend API is online');
    } catch (error) {
      setApiStatus('offline');
      console.warn('⚠️ Backend API is offline - Demo mode available');
      console.log('To start the backend: cd backend && python main.py');
    }
  };

  const handleUserSelection = (userType: 'doctor' | 'patient') => {
    const newUser: User = {
      id: `${userType}_${Date.now()}`,
      type: userType,
      name: userType === 'doctor' ? 'Dr. Smith' : 'Patient',
      email: `${userType}@example.com`,
    };
    setUser(newUser);
  };

  const handleLogout = () => {
    setUser(null);
  };

  if (isLoading) {
    return <LoadingScreen />;
  }

  return (
    <Router>
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-cyan-50">
        <AnimatePresence mode="wait">
          {!user ? (
            showTroubleshooting ? (
              <motion.div
                key="troubleshooting"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="min-h-screen flex items-center justify-center p-6"
              >
                <div className="w-full">
                  <TroubleshootingGuide onRetry={() => {
                    setShowTroubleshooting(false);
                    checkApiHealth();
                  }} />
                  <div className="text-center mt-6">
                    <button
                      onClick={() => setShowTroubleshooting(false)}
                      className="btn-secondary"
                    >
                      Continue with Demo Mode
                    </button>
                  </div>
                </div>
              </motion.div>
            ) : (
              <UserSelectionScreen
                key="selection"
                onUserSelect={handleUserSelection}
                apiStatus={apiStatus}
                onShowTroubleshooting={() => setShowTroubleshooting(true)}
              />
            )
          ) : (
            <Routes>
              <Route
                path="/doctor"
                element={
                  user.type === 'doctor' ? (
                    <DoctorInterface user={user} onLogout={handleLogout} />
                  ) : (
                    <Navigate to="/patient" replace />
                  )
                }
              />
              <Route
                path="/patient"
                element={
                  user.type === 'patient' ? (
                    <PatientInterface user={user} onLogout={handleLogout} />
                  ) : (
                    <Navigate to="/doctor" replace />
                  )
                }
              />
              <Route
                path="/"
                element={
                  <Navigate
                    to={user.type === 'doctor' ? '/doctor' : '/patient'}
                    replace
                  />
                }
              />
            </Routes>
          )}
        </AnimatePresence>
      </div>
    </Router>
  );
}

const LoadingScreen: React.FC = () => (
  <motion.div
    initial={{ opacity: 0 }}
    animate={{ opacity: 1 }}
    exit={{ opacity: 0 }}
    className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 via-white to-cyan-50"
  >
    <div className="text-center">
      <motion.div
        animate={{ rotate: 360 }}
        transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
        className="w-16 h-16 mx-auto mb-6"
      >
        <Activity className="w-full h-full text-primary-600" />
      </motion.div>
      <motion.h1
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.5 }}
        className="text-3xl font-bold text-medical-800 mb-2"
      >
        Uterine Fibroids Analyzer
      </motion.h1>
      <motion.p
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.7 }}
        className="text-medical-600"
      >
        AI-Powered Medical Imaging Analysis
      </motion.p>
    </div>
  </motion.div>
);

interface UserSelectionScreenProps {
  onUserSelect: (userType: 'doctor' | 'patient') => void;
  apiStatus: 'checking' | 'online' | 'offline';
  onShowTroubleshooting: () => void;
}

const UserSelectionScreen: React.FC<UserSelectionScreenProps> = ({
  onUserSelect,
  apiStatus,
  onShowTroubleshooting
}) => (
  <motion.div
    initial={{ opacity: 0 }}
    animate={{ opacity: 1 }}
    exit={{ opacity: 0 }}
    className="min-h-screen flex items-center justify-center p-6"
  >
    <div className="max-w-4xl w-full">
      {/* Header */}
      <motion.div
        initial={{ y: -50, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.2 }}
        className="text-center mb-12"
      >
        <div className="flex items-center justify-center mb-6">
          <Activity className="w-12 h-12 text-primary-600 mr-3" />
          <h1 className="text-4xl font-bold text-medical-800">
            Uterine Fibroids Analyzer
          </h1>
        </div>
        <p className="text-xl text-medical-600 mb-4">
          Advanced AI-powered analysis for uterine fibroids detection
        </p>
        <div className="flex items-center justify-center space-x-4">
          <div className="flex items-center space-x-2">
            <div className={`w-3 h-3 rounded-full ${apiStatus === 'online' ? 'bg-green-500' :
              apiStatus === 'offline' ? 'bg-red-500' : 'bg-yellow-500'
              }`} />
            <span className="text-sm text-medical-500">
              {apiStatus === 'online' ? 'API Online' :
                apiStatus === 'offline' ? 'Demo Mode' : 'Checking API...'}
            </span>
          </div>
          {apiStatus === 'offline' && (
            <button
              onClick={onShowTroubleshooting}
              className="flex items-center space-x-1 text-xs text-red-600 hover:text-red-700 transition-colors"
            >
              <AlertTriangle className="w-3 h-3" />
              <span>Need Help?</span>
            </button>
          )}
        </div>
      </motion.div>

      {/* User Selection Cards */}
      <div className="grid md:grid-cols-2 gap-8">
        <motion.div
          initial={{ x: -100, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ delay: 0.4 }}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={() => onUserSelect('doctor')}
          className="medical-glass p-8 cursor-pointer group hover:shadow-2xl transition-all duration-300"
        >
          <div className="text-center">
            <div className="w-20 h-20 mx-auto mb-6 bg-gradient-to-br from-primary-500 to-primary-600 rounded-2xl flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
              <Stethoscope className="w-10 h-10 text-white" />
            </div>
            <h2 className="text-2xl font-bold text-medical-800 mb-4">
              Doctor Interface
            </h2>
            <p className="text-medical-600 mb-6">
              Upload medical scans, analyze with AI, and generate comprehensive reports with explainable AI insights.
            </p>
            <div className="space-y-2 text-sm text-medical-500">
              <div className="flex items-center justify-center">
                <Shield className="w-4 h-4 mr-2" />
                <span>STEP 1: Upload Scan</span>
              </div>
              <div className="flex items-center justify-center">
                <Activity className="w-4 h-4 mr-2" />
                <span>STEP 2: AI Analysis</span>
              </div>
              <div className="flex items-center justify-center">
                <Users className="w-4 h-4 mr-2" />
                <span>STEP 3: XAI Explanation</span>
              </div>
              <div className="flex items-center justify-center">
                <Stethoscope className="w-4 h-4 mr-2" />
                <span>STEP 4: Generate Report</span>
              </div>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ x: 100, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ delay: 0.6 }}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={() => onUserSelect('patient')}
          className="medical-glass p-8 cursor-pointer group hover:shadow-2xl transition-all duration-300"
        >
          <div className="text-center">
            <div className="w-20 h-20 mx-auto mb-6 bg-gradient-to-br from-medical-500 to-medical-600 rounded-2xl flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
              <Users className="w-10 h-10 text-white" />
            </div>
            <h2 className="text-2xl font-bold text-medical-800 mb-4">
              Patient Interface
            </h2>
            <p className="text-medical-600 mb-6">
              Upload your scan, get AI analysis results, and receive personalized health insights through our interactive chat.
            </p>
            <div className="space-y-2 text-sm text-medical-500">
              <div className="flex items-center justify-center">
                <Shield className="w-4 h-4 mr-2" />
                <span>STEP 1: Upload Scan</span>
              </div>
              <div className="flex items-center justify-center">
                <Activity className="w-4 h-4 mr-2" />
                <span>STEP 2: AI Analysis</span>
              </div>
              <div className="flex items-center justify-center">
                <Users className="w-4 h-4 mr-2" />
                <span>STEP 3: Health Chat</span>
              </div>
              <div className="flex items-center justify-center">
                <Stethoscope className="w-4 h-4 mr-2" />
                <span>STEP 4: Download Report</span>
              </div>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Footer */}
      <motion.div
        initial={{ y: 50, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.8 }}
        className="text-center mt-12"
      >
        <p className="text-medical-500 text-sm">
          Powered by U-Net++ Deep Learning • Explainable AI • HIPAA Compliant
        </p>
      </motion.div>
    </div>
  </motion.div>
);

export default App;
