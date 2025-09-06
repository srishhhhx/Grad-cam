import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  User,
  Activity,
  CheckCircle,
  ArrowRight,
  FileText
} from 'lucide-react';
import { UNetPrediction } from '../types';

export interface QuestionnaireData {
  demographics: {
    age: number | null;
    sex: string;
    ethnicity: string;
  };
  symptoms: {
    heavy_bleeding: string;
    pelvic_pain: string;
    other_symptoms: string;
  };
  medical_history: {
    family_history_fibroids: string;
    previous_pregnancies: string;
    hormonal_medications: string;
  };
}

interface QuestionnaireFormProps {
  prediction: UNetPrediction;
  onSubmit: (data: QuestionnaireData) => void;
  isLoading?: boolean;
}

const QuestionnaireForm: React.FC<QuestionnaireFormProps> = ({
  prediction: _prediction,
  onSubmit,
  isLoading = false
}) => {
  const [formData, setFormData] = useState<QuestionnaireData>({
    demographics: {
      age: null,
      sex: '',
      ethnicity: ''
    },
    symptoms: {
      heavy_bleeding: '',
      pelvic_pain: '',
      other_symptoms: ''
    },
    medical_history: {
      family_history_fibroids: '',
      previous_pregnancies: '',
      hormonal_medications: ''
    }
  });

  const [errors, setErrors] = useState<Record<string, string>>({});
  const [currentSection, setCurrentSection] = useState(0);

  const sections = [
    {
      title: 'Personal Information',
      icon: User,
      fields: ['demographics']
    },
    {
      title: 'Current Symptoms',
      icon: Activity,
      fields: ['symptoms']
    },
    {
      title: 'Medical History',
      icon: FileText,
      fields: ['medical_history']
    }
  ];

  const validateForm = (): boolean => {
    const newErrors: Record<string, string> = {};

    // Required fields validation
    if (!formData.demographics.age || formData.demographics.age < 18 || formData.demographics.age > 100) {
      newErrors['demographics.age'] = 'Please enter a valid age between 18 and 100';
    }

    if (!formData.demographics.sex) {
      newErrors['demographics.sex'] = 'Please select your biological sex';
    }

    if (!formData.symptoms.heavy_bleeding) {
      newErrors['symptoms.heavy_bleeding'] = 'Please answer this question';
    }

    if (!formData.symptoms.pelvic_pain) {
      newErrors['symptoms.pelvic_pain'] = 'Please answer this question';
    }

    if (!formData.medical_history.family_history_fibroids) {
      newErrors['medical_history.family_history_fibroids'] = 'Please answer this question';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (validateForm()) {
      onSubmit(formData);
    }
  };

  const updateField = (section: keyof QuestionnaireData, field: string, value: any) => {
    setFormData(prev => ({
      ...prev,
      [section]: {
        ...prev[section],
        [field]: value
      }
    }));

    // Clear error when user starts typing
    const errorKey = `${section}.${field}`;
    if (errors[errorKey]) {
      setErrors(prev => {
        const newErrors = { ...prev };
        delete newErrors[errorKey];
        return newErrors;
      });
    }
  };

  const nextSection = () => {
    if (currentSection < sections.length - 1) {
      setCurrentSection(currentSection + 1);
    }
  };

  const prevSection = () => {
    if (currentSection > 0) {
      setCurrentSection(currentSection - 1);
    }
  };

  const renderDemographicsSection = () => (
    <div className="space-y-6">
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Age *
        </label>
        <input
          type="number"
          min="18"
          max="100"
          value={formData.demographics.age || ''}
          onChange={(e) => updateField('demographics', 'age', parseInt(e.target.value) || null)}
          className={`input-glass w-full ${errors['demographics.age'] ? 'border-red-400' : ''}`}
          placeholder="Enter your age"
        />
        {errors['demographics.age'] && (
          <p className="text-red-400 text-sm mt-1">{errors['demographics.age']}</p>
        )}
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Biological Sex *
        </label>
        <div className="space-y-2">
          {['Female', 'Male'].map((option) => (
            <label key={option} className="flex items-center space-x-3 cursor-pointer">
              <input
                type="radio"
                name="sex"
                value={option}
                checked={formData.demographics.sex === option}
                onChange={(e) => updateField('demographics', 'sex', e.target.value)}
                className="w-4 h-4 text-medical-400 bg-transparent border-gray-400 focus:ring-medical-400"
              />
              <span className="text-gray-300">{option}</span>
            </label>
          ))}
        </div>
        {errors['demographics.sex'] && (
          <p className="text-red-400 text-sm mt-1">{errors['demographics.sex']}</p>
        )}
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Ethnicity (Optional)
        </label>
        <input
          type="text"
          value={formData.demographics.ethnicity}
          onChange={(e) => updateField('demographics', 'ethnicity', e.target.value)}
          className="input-glass w-full"
          placeholder="e.g., Hispanic, African American, Caucasian, Asian, etc."
        />
      </div>
    </div>
  );

  const renderSymptomsSection = () => (
    <div className="space-y-6">
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Do you experience heavy menstrual bleeding? *
        </label>
        <div className="space-y-2">
          {['Yes', 'No'].map((option) => (
            <label key={option} className="flex items-center space-x-3 cursor-pointer">
              <input
                type="radio"
                name="heavy_bleeding"
                value={option}
                checked={formData.symptoms.heavy_bleeding === option}
                onChange={(e) => updateField('symptoms', 'heavy_bleeding', e.target.value)}
                className="w-4 h-4 text-medical-400 bg-transparent border-gray-400 focus:ring-medical-400"
              />
              <span className="text-gray-300">{option}</span>
            </label>
          ))}
        </div>
        {errors['symptoms.heavy_bleeding'] && (
          <p className="text-red-400 text-sm mt-1">{errors['symptoms.heavy_bleeding']}</p>
        )}
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Do you experience pelvic pain or pressure? *
        </label>
        <div className="space-y-2">
          {['Yes', 'No'].map((option) => (
            <label key={option} className="flex items-center space-x-3 cursor-pointer">
              <input
                type="radio"
                name="pelvic_pain"
                value={option}
                checked={formData.symptoms.pelvic_pain === option}
                onChange={(e) => updateField('symptoms', 'pelvic_pain', e.target.value)}
                className="w-4 h-4 text-medical-400 bg-transparent border-gray-400 focus:ring-medical-400"
              />
              <span className="text-gray-300">{option}</span>
            </label>
          ))}
        </div>
        {errors['symptoms.pelvic_pain'] && (
          <p className="text-red-400 text-sm mt-1">{errors['symptoms.pelvic_pain']}</p>
        )}
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Other symptoms (Optional)
        </label>
        <textarea
          value={formData.symptoms.other_symptoms}
          onChange={(e) => updateField('symptoms', 'other_symptoms', e.target.value)}
          className="input-glass w-full h-24 resize-none"
          placeholder="Please describe any other symptoms you're experiencing..."
        />
      </div>
    </div>
  );

  const renderMedicalHistorySection = () => (
    <div className="space-y-6">
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Do you have a family history of uterine fibroids? *
        </label>
        <div className="space-y-2">
          {['Yes', 'No', 'Not sure'].map((option) => (
            <label key={option} className="flex items-center space-x-3 cursor-pointer">
              <input
                type="radio"
                name="family_history_fibroids"
                value={option}
                checked={formData.medical_history.family_history_fibroids === option}
                onChange={(e) => updateField('medical_history', 'family_history_fibroids', e.target.value)}
                className="w-4 h-4 text-medical-400 bg-transparent border-gray-400 focus:ring-medical-400"
              />
              <span className="text-gray-300">{option}</span>
            </label>
          ))}
        </div>
        {errors['medical_history.family_history_fibroids'] && (
          <p className="text-red-400 text-sm mt-1">{errors['medical_history.family_history_fibroids']}</p>
        )}
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Have you been pregnant before? (Optional)
        </label>
        <div className="space-y-2">
          {['Yes', 'No', 'Prefer not to say'].map((option) => (
            <label key={option} className="flex items-center space-x-3 cursor-pointer">
              <input
                type="radio"
                name="previous_pregnancies"
                value={option}
                checked={formData.medical_history.previous_pregnancies === option}
                onChange={(e) => updateField('medical_history', 'previous_pregnancies', e.target.value)}
                className="w-4 h-4 text-medical-400 bg-transparent border-gray-400 focus:ring-medical-400"
              />
              <span className="text-gray-300">{option}</span>
            </label>
          ))}
        </div>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Are you currently taking hormonal medications? (Optional)
        </label>
        <div className="space-y-2">
          {['Yes', 'No', 'Prefer not to say'].map((option) => (
            <label key={option} className="flex items-center space-x-3 cursor-pointer">
              <input
                type="radio"
                name="hormonal_medications"
                value={option}
                checked={formData.medical_history.hormonal_medications === option}
                onChange={(e) => updateField('medical_history', 'hormonal_medications', e.target.value)}
                className="w-4 h-4 text-medical-400 bg-transparent border-gray-400 focus:ring-medical-400"
              />
              <span className="text-gray-300">{option}</span>
            </label>
          ))}
        </div>
      </div>
    </div>
  );

  return (
    <div className="max-w-4xl mx-auto p-6">
      {/* Header */}
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold text-white mb-4">Health Questionnaire</h2>
        <p className="text-gray-300 text-lg">
          Please fill out this form to help us provide personalized guidance about your health.
        </p>
      </div>

      {/* Progress Indicator */}
      <div className="flex justify-center mb-8">
        <div className="flex space-x-4">
          {sections.map((section, index) => {
            const Icon = section.icon;
            const isActive = index === currentSection;
            const isCompleted = index < currentSection;

            return (
              <div key={index} className="flex items-center">
                <div
                  className={`w-12 h-12 rounded-full flex items-center justify-center border-2 transition-all duration-300 ${isActive
                    ? 'border-medical-400 bg-medical-400/20 text-medical-400'
                    : isCompleted
                      ? 'border-green-400 bg-green-400/20 text-green-400'
                      : 'border-gray-500 text-gray-500'
                    }`}
                >
                  {isCompleted ? <CheckCircle className="w-6 h-6" /> : <Icon className="w-6 h-6" />}
                </div>
                {index < sections.length - 1 && (
                  <div
                    className={`w-8 h-0.5 mx-2 transition-all duration-300 ${isCompleted ? 'bg-green-400' : 'bg-gray-500'
                      }`}
                  />
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Form Content */}
      <motion.div
        key={currentSection}
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        exit={{ opacity: 0, x: -20 }}
        transition={{ duration: 0.3 }}
        className="glass-card p-8"
      >
        <div className="flex items-center mb-6">
          {React.createElement(sections[currentSection].icon, {
            className: "w-8 h-8 text-medical-400 mr-3"
          })}
          <h3 className="text-2xl font-semibold text-white">
            {sections[currentSection].title}
          </h3>
        </div>

        <form onSubmit={handleSubmit}>
          {currentSection === 0 && renderDemographicsSection()}
          {currentSection === 1 && renderSymptomsSection()}
          {currentSection === 2 && renderMedicalHistorySection()}

          {/* Navigation Buttons */}
          <div className="flex justify-between mt-8">
            <button
              type="button"
              onClick={prevSection}
              disabled={currentSection === 0}
              className={`btn-secondary ${currentSection === 0 ? 'opacity-50 cursor-not-allowed' : ''
                }`}
            >
              Previous
            </button>

            {currentSection < sections.length - 1 ? (
              <button
                type="button"
                onClick={nextSection}
                className="btn-primary"
              >
                Next
                <ArrowRight className="w-4 h-4 ml-2" />
              </button>
            ) : (
              <button
                type="submit"
                disabled={isLoading}
                className="btn-primary"
              >
                {isLoading ? 'Processing...' : 'Start Chat'}
                <ArrowRight className="w-4 h-4 ml-2" />
              </button>
            )}
          </div>
        </form>
      </motion.div>
    </div>
  );
};

export default QuestionnaireForm;
