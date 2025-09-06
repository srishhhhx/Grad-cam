import React from 'react';
import { motion } from 'framer-motion';
import { 
  AlertTriangle, 
  Server, 
  Terminal, 
  CheckCircle, 
  ExternalLink,
  RefreshCw
} from 'lucide-react';

interface TroubleshootingGuideProps {
  onRetry: () => void;
}

const TroubleshootingGuide: React.FC<TroubleshootingGuideProps> = ({ onRetry }) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="medical-glass p-6 max-w-2xl mx-auto"
    >
      <div className="flex items-center space-x-3 mb-6">
        <div className="w-12 h-12 bg-yellow-100 rounded-full flex items-center justify-center">
          <AlertTriangle className="w-6 h-6 text-yellow-600" />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-medical-800">
            Backend Connection Issue
          </h3>
          <p className="text-medical-600">
            The backend server is not responding. Here's how to fix it:
          </p>
        </div>
      </div>

      <div className="space-y-4">
        <div className="bg-blue-50/30 border border-blue-200/30 rounded-xl p-4">
          <div className="flex items-start space-x-3">
            <div className="w-6 h-6 bg-blue-100 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
              <Server className="w-4 h-4 text-blue-600" />
            </div>
            <div>
              <h4 className="font-medium text-blue-800 mb-2">
                1. Start the Backend Server
              </h4>
              <div className="bg-gray-900 text-green-400 p-3 rounded-lg font-mono text-sm mb-3">
                <div># Navigate to backend directory</div>
                <div>cd backend</div>
                <div className="mt-1"># Activate virtual environment (if created)</div>
                <div>source venv/bin/activate  # Mac/Linux</div>
                <div>venv\Scripts\activate     # Windows</div>
                <div className="mt-1"># Start the server</div>
                <div>python main.py</div>
              </div>
              <p className="text-blue-700 text-sm">
                The backend should start on <strong>http://localhost:8000</strong>
              </p>
            </div>
          </div>
        </div>

        <div className="bg-green-50/30 border border-green-200/30 rounded-xl p-4">
          <div className="flex items-start space-x-3">
            <div className="w-6 h-6 bg-green-100 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
              <Terminal className="w-4 h-4 text-green-600" />
            </div>
            <div>
              <h4 className="font-medium text-green-800 mb-2">
                2. Quick Setup (First Time)
              </h4>
              <div className="bg-gray-900 text-green-400 p-3 rounded-lg font-mono text-sm mb-3">
                <div># Install dependencies first</div>
                <div>cd backend</div>
                <div>python -m venv venv</div>
                <div>source venv/bin/activate</div>
                <div>pip install -r requirements.txt</div>
                <div>python main.py</div>
              </div>
              <p className="text-green-700 text-sm">
                This creates a virtual environment and installs all required packages.
              </p>
            </div>
          </div>
        </div>

        <div className="bg-purple-50/30 border border-purple-200/30 rounded-xl p-4">
          <div className="flex items-start space-x-3">
            <div className="w-6 h-6 bg-purple-100 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
              <CheckCircle className="w-4 h-4 text-purple-600" />
            </div>
            <div>
              <h4 className="font-medium text-purple-800 mb-2">
                3. Verify Backend is Running
              </h4>
              <p className="text-purple-700 text-sm mb-3">
                Once the backend is running, you should see:
              </p>
              <div className="bg-gray-900 text-green-400 p-3 rounded-lg font-mono text-sm mb-3">
                <div>🚀 Starting Uterine Fibroids Analyzer API...</div>
                <div>📊 Loading U-Net model...</div>
                <div>🔬 Loading XAI analysis engine...</div>
                <div>📄 Initializing PDF generator...</div>
                <div>✅ API startup complete!</div>
                <div>INFO: Uvicorn running on http://0.0.0.0:8000</div>
              </div>
              <div className="flex items-center space-x-2">
                <span className="text-purple-700 text-sm">Test the API:</span>
                <a
                  href="http://localhost:8000/api/health"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center space-x-1 text-purple-600 hover:text-purple-700 text-sm"
                >
                  <span>http://localhost:8000/api/health</span>
                  <ExternalLink className="w-3 h-3" />
                </a>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-amber-50/30 border border-amber-200/30 rounded-xl p-4">
          <div className="flex items-start space-x-3">
            <div className="w-6 h-6 bg-amber-100 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
              <AlertTriangle className="w-4 h-4 text-amber-600" />
            </div>
            <div>
              <h4 className="font-medium text-amber-800 mb-2">
                Demo Mode Available
              </h4>
              <p className="text-amber-700 text-sm">
                Don't worry! The application includes a <strong>demo mode</strong> that works 
                without the backend. You can still:
              </p>
              <ul className="list-disc list-inside text-amber-700 text-sm mt-2 space-y-1">
                <li>Upload and preview medical images</li>
                <li>See mock AI predictions and analysis</li>
                <li>Experience the full UI workflow</li>
                <li>Generate sample PDF reports</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      <div className="flex items-center justify-between mt-6 pt-4 border-t border-white/20">
        <div className="text-sm text-medical-600">
          <p>Need more help? Check the <strong>README_WEBAPP.md</strong> file</p>
        </div>
        <button
          onClick={onRetry}
          className="btn-primary flex items-center space-x-2"
        >
          <RefreshCw className="w-4 h-4" />
          <span>Retry Connection</span>
        </button>
      </div>
    </motion.div>
  );
};

export default TroubleshootingGuide;
