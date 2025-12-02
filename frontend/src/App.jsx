import { useState, useRef } from 'react';
import { 
  Upload, Activity, AlertCircle, CheckCircle, XCircle, 
  BarChart3, Image as ImageIcon, Info, Sparkles, Shield 
} from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const API_URL = 'http://localhost:8000';

export default function SkinFusionAnalyzer() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleImageSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (file.size > 10 * 1024 * 1024) {
        setError({ type: 'general', message: 'File size must be less than 10MB' });
        return;
      }
      
      if (!file.type.startsWith('image/')) {
        setError({ type: 'general', message: 'Please select a valid image file' });
        return;
      }
      
      setImage(file);
      const objectUrl = URL.createObjectURL(file);
      setPreview(objectUrl);
      setResult(null);
      setError(null);
    }
  };

  const handleAnalyze = async () => {
    if (!image) return;

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', image);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        
        if (response.status === 400 && errorData.detail) {
          if (typeof errorData.detail === 'object' && errorData.detail.error === 'Invalid image type') {
            setError({
              type: 'invalid_image',
              message: errorData.detail.message,
              skin_probability: errorData.detail.skin_probability
            });
          } else if (typeof errorData.detail === 'string') {
            setError({ type: 'general', message: errorData.detail });
          } else {
            setError({ type: 'general', message: 'Invalid image format' });
          }
        } else {
          setError({ type: 'general', message: 'Prediction failed. Please try again.' });
        }
        return;
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError({ 
        type: 'connection', 
        message: 'Failed to connect to server. Make sure the backend is running on port 8000.' 
      });
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const getSeverityIcon = (severity) => {
    const icons = {
      'Critical': <XCircle className="w-5 h-5" />,
      'High': <AlertCircle className="w-5 h-5" />,
      'Medium': <AlertCircle className="w-5 h-5" />,
      'Low': <CheckCircle className="w-5 h-5" />
    };
    return icons[severity] || icons.Low;
  };

  const getSeverityStyles = (severity) => {
    const styles = {
      'Critical': 'bg-purple-500 text-white',
      'High': 'bg-red-500 text-white',
      'Medium': 'bg-orange-500 text-white',
      'Low': 'bg-green-500 text-white'
    };
    return styles[severity] || styles.Low;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      <header className="sticky top-0 z-50 backdrop-blur-xl bg-white/80 border-b border-gray-200/50 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="relative">
                <Activity className="w-8 h-8 text-blue-600" />
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-500 rounded-full border-2 border-white"></div>
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                  SkinFusion Analyzer
                </h1>
                <p className="text-sm text-gray-600">AI-Powered Dermatology Assistant</p>
              </div>
            </div>
            <div className="hidden sm:flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-full border border-blue-200">
              <Shield className="w-4 h-4 text-blue-600" />
              <span className="text-sm font-medium text-blue-700">Validated Detection</span>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 lg:gap-8">
          {/* LEFT PANEL – Upload + Errors + About */}
          <div className="space-y-6">
            {/* Upload Card */}
            <div className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-xl border border-gray-200/50 overflow-hidden transition-all hover:shadow-2xl">
              <div className="bg-gradient-to-r from-blue-600 to-indigo-600 p-6">
                <h2 className="text-xl font-bold text-white flex items-center gap-2">
                  <Upload className="w-6 h-6" />
                  Upload Skin Lesion Image
                </h2>
              </div>

              <div className="p-6 space-y-4">
                <div 
                  className="border-2 border-dashed rounded-2xl p-4 flex flex-col items-center justify-center bg-slate-50/80 hover:bg-slate-100 cursor-pointer transition-all"
                  onClick={() => fileInputRef.current?.click()}
                >
                  {preview ? (
                    <div className="w-full rounded-2xl overflow-hidden border border-gray-200/80 shadow-inner">
                      <img 
                        src={preview} 
                        alt="Preview" 
                        className="w-full h-64 object-cover"
                      />
                    </div>
                  ) : (
                    <div className="flex flex-col items-center gap-3 py-10">
                      <div className="p-4 bg-blue-50 rounded-full">
                        <ImageIcon className="w-8 h-8 text-blue-600" />
                      </div>
                      <div className="text-center space-y-1">
                        <p className="font-semibold text-gray-800">Click to upload skin lesion image</p>
                        <p className="text-xs text-gray-500">
                          PNG, JPG, JPEG • Max 10MB • Prefer dermatoscopic images
                        </p>
                      </div>
                    </div>
                  )}
                  <input 
                    type="file" 
                    accept="image/*" 
                    className="hidden" 
                    ref={fileInputRef}
                    onChange={handleImageSelect}
                  />
                </div>

                <button
                  onClick={handleAnalyze}
                  disabled={!image || loading}
                  className={`w-full flex items-center justify-center gap-2 py-3 rounded-xl font-semibold text-white text-sm
                    ${image && !loading 
                      ? 'bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 shadow-lg hover:shadow-xl' 
                      : 'bg-gray-400 cursor-not-allowed'}
                    transition-all`}
                >
                  {loading ? (
                    <>
                      <Sparkles className="w-4 h-4 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Activity className="w-4 h-4" />
                      Analyze Lesion
                    </>
                  )}
                </button>

                {/* Error Messages */}
                {error && (
                  <div className="bg-red-50 border border-red-200 rounded-2xl p-4 flex gap-3">
                    <AlertCircle className="w-5 h-5 text-red-600 mt-0.5" />
                    <div>
                      <p className="font-semibold text-red-800 mb-1">
                        {error.type === 'invalid_image' ? '⚠️ Invalid Image Type' : '❌ Error'}
                      </p>
                      <p className="text-sm text-red-800">
                        {error.message || error}
                      </p>
                      {error.type === 'invalid_image' && error.skin_probability !== undefined && (
                        <p className="text-xs text-red-700 mt-2">
                          Skin detection confidence: {error.skin_probability}%
                        </p>
                      )}
                      {error.type === 'invalid_image' && (
                        <div className="mt-3 p-3 bg-white/60 rounded-lg">
                          <p className="text-xs font-semibold text-red-900 mb-1">
                            Please ensure:
                          </p>
                          <ul className="text-xs text-red-800 space-y-1">
                            <li>• Image shows a skin lesion or mole</li>
                            <li>• Photo is clear and well-lit</li>
                            <li>• Lesion is the main focus of the image</li>
                          </ul>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* About Tool */}
            <div className="bg-gradient-to-br from-blue-600 to-indigo-600 rounded-3xl p-6 shadow-xl text-white">
              <div className="flex items-center gap-2 mb-4">
                <Info className="w-6 h-6" />
                <h3 className="font-bold text-lg">About This Tool</h3>
              </div>
              <div className="space-y-3">
                <div className="flex items-start gap-3 bg-white/10 backdrop-blur-sm rounded-xl p-3">
                  <Shield className="w-5 h-5 flex-shrink-0 mt-0.5" />
                  <p className="text-sm">AI-powered validation ensures uploaded images look like skin lesions.</p>
                </div>
                <div className="flex items-start gap-3 bg-white/10 backdrop-blur-sm rounded-xl p-3">
                  <CheckCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
                  <p className="text-sm">Advanced ensemble of ConvNeXt, EfficientNet & ResNet models.</p>
                </div>
                <div className="flex items-start gap-3 bg-white/10 backdrop-blur-sm rounded-xl p-3">
                  <CheckCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
                  <p className="text-sm">Trained on 10,000+ dermatoscopic images.</p>
                </div>
                <div className="flex items-start gap-3 bg-white/10 backdrop-blur-sm rounded-xl p-3">
                  <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
                  <p className="text-sm font-semibold">
                    Educational purposes only – always consult a dermatologist.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* RIGHT PANEL – Results / Placeholder */}
          <div className="space-y-6">
            {result ? (
              <>
                {/* Diagnosis Result */}
                <div className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-xl border border-gray-200/50 overflow-hidden">
                  <div className="bg-gradient-to-r from-indigo-600 to-purple-600 p-6">
                    <h2 className="text-xl font-bold text-white flex items-center gap-2">
                      <BarChart3 className="w-6 h-6" />
                      Diagnosis Result
                      {result.skin_validated && (
                        <span className="ml-auto flex items-center gap-1 text-xs bg-white/20 px-2 py-1 rounded-full">
                          <Shield className="w-3 h-3" />
                          Validated
                        </span>
                      )}
                    </h2>
                  </div>

                  <div className="p-6 space-y-4">
                    <div
                      className="rounded-2xl p-6 border-2"
                      style={{ 
                        backgroundColor: `${result.color}15`,
                        borderColor: `${result.color}40`
                      }}
                    >
                      <div className="flex items-start justify-between mb-4">
                        <div>
                          <h3 className="text-3xl font-bold text-gray-900 mb-2">
                            {result.class_name}
                          </h3>
                          <span className="inline-block px-3 py-1 bg-white/80 rounded-full text-xs font-mono font-semibold text-gray-700">
                            {result.class_code?.toUpperCase() || 'N/A'}
                          </span>
                        </div>
                        <div className={`p-3 rounded-full ${getSeverityStyles(result.severity)}`}>
                          {getSeverityIcon(result.severity)}
                        </div>
                      </div>

                      <p className="text-gray-700 mb-4 leading-relaxed">{result.description}</p>

                      <div className="bg-white/90 backdrop-blur-sm rounded-xl p-4 shadow-sm">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-semibold text-gray-700">
                            Confidence Level
                          </span>
                          <span className="text-3xl font-bold" style={{ color: result.color }}>
                            {result.confidence}%
                          </span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                          <div
                            className="h-full rounded-full transition-all duration-1000 ease-out"
                            style={{ 
                              width: `${result.confidence}%`,
                              backgroundColor: result.color
                            }}
                          />
                        </div>
                      </div>

                      {/* Low-confidence warning */}
                      {result.confidence_flag === 'low' && (
                        <div className="mt-4 flex items-start gap-2 bg-amber-50 border border-amber-200 rounded-xl p-3">
                          <AlertCircle className="w-4 h-4 text-amber-700 mt-0.5" />
                          <p className="text-xs text-amber-900 leading-relaxed">
                            This prediction has <span className="font-semibold">low confidence</span> (
                            {result.confidence}% &lt; {result.confidence_threshold}%). The image may be unclear,
                            poorly lit, or the lesion may not be the main focus. Try uploading a clearer 
                            close-up and always consult a dermatologist for medical decisions.
                          </p>
                        </div>
                      )}
                    </div>

                    {/* Risk badge with "uncertain" wording */}
                    <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-full font-semibold text-sm ${getSeverityStyles(result.severity)} shadow-lg`}>
                      {getSeverityIcon(result.severity)}
                      {result.prediction_status === 'uncertain'
                        ? `Uncertain – Possible ${result.severity} Risk`
                        : `${result.severity} Risk Level`}
                    </div>
                  </div>
                </div>

                {/* Probability Distribution */}
                <div className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-xl border border-gray-200/50 overflow-hidden">
                  <div className="p-6 border-b border-gray-200">
                    <h3 className="text-lg font-bold text-gray-900">
                      Probability Distribution
                    </h3>
                  </div>
                  
                  <div className="p-6">
                    <div className="w-full h-72 mb-6">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={result.all_probabilities}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                          <XAxis 
                            dataKey="class_name" 
                            angle={-45}
                            textAnchor="end"
                            height={100}
                            tick={{ fontSize: 11, fill: '#6b7280' }}
                          />
                          <YAxis 
                            label={{ 
                              value: 'Probability (%)', 
                              angle: -90, 
                              position: 'insideLeft',
                              style: { fontSize: 12, fill: '#6b7280' }
                            }}
                            tick={{ fontSize: 11, fill: '#6b7280' }}
                          />
                          <Tooltip 
                            formatter={(value) => `${value}%`}
                            contentStyle={{ 
                              borderRadius: '12px', 
                              border: '1px solid #e5e7eb',
                              boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'
                            }}
                          />
                          <Bar dataKey="probability" radius={[8, 8, 0, 0]}>
                            {result.all_probabilities.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={entry.color} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>

                    <div className="space-y-3">
                      <p className="text-sm font-bold text-gray-700 mb-3">Top 3 Predictions:</p>
                      {result.all_probabilities.slice(0, 3).map((item, idx) => (
                        <div 
                          key={idx}
                          className="flex items-center justify-between p-4 bg-gradient-to-r from-gray-50 to-white rounded-xl border border-gray-200 hover:shadow-md transition-all"
                        >
                          <div className="flex items-center gap-3">
                            <div 
                              className="w-4 h-4 rounded-full shadow-sm"
                              style={{ backgroundColor: item.color }}
                            />
                            <span className="text-sm font-semibold text-gray-800">
                              {item.class_name}
                            </span>
                          </div>
                          <span className="text-lg font-bold" style={{ color: item.color }}>
                            {item.probability}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Disclaimer */}
                <div className="bg-amber-50 border-2 border-amber-300 rounded-2xl p-5 shadow-lg">
                  <div className="flex gap-3">
                    <AlertCircle className="w-6 h-6 text-amber-600 flex-shrink-0" />
                    <div className="text-sm text-amber-900">
                      <p className="font-bold mb-2">⚠️ Medical Disclaimer</p>
                      <p className="leading-relaxed">
                        This AI tool is for educational purposes only. Always consult a qualified 
                        dermatologist for proper medical diagnosis and treatment.
                      </p>
                    </div>
                  </div>
                </div>
              </>
            ) : (
              <>
                {/* Placeholder when no result */}
                <div className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-xl border border-gray-200/50 p-12 text-center">
                  <div className="inline-block p-6 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-full mb-6">
                    <Activity className="w-16 h-16 text-blue-600" />
                  </div>
                  <h3 className="text-xl font-bold text-gray-900 mb-2">
                    Ready to Analyze
                  </h3>
                  <p className="text-gray-600">
                    Upload a skin lesion image to get started with AI-powered analysis.
                  </p>
                </div>

                <div className="bg-gradient-to-br from-indigo-600 to-purple-600 rounded-3xl p-6 shadow-xl text-white">
                  <h3 className="font-bold text-lg mb-4 flex items-center gap-2">
                    <BarChart3 className="w-6 h-6" />
                    How It Works
                  </h3>
                  <div className="space-y-4">
                    <div className="flex gap-4">
                      <div className="w-8 h-8 rounded-full bg-white/20 flex items-center justify-center font-bold flex-shrink-0">1</div>
                      <div>
                        <p className="font-semibold mb-1">Upload Image</p>
                        <p className="text-sm text-white/90">Select a clear dermatoscopic image of the skin lesion.</p>
                      </div>
                    </div>
                    <div className="flex gap-4">
                      <div className="w-8 h-8 rounded-full bg-white/20 flex items-center justify-center font-bold flex-shrink-0">2</div>
                      <div>
                        <p className="font-semibold mb-1">Validation</p>
                        <p className="text-sm text-white/90">Gemini confirms that the image looks like a real skin lesion.</p>
                      </div>
                    </div>
                    <div className="flex gap-4">
                      <div className="w-8 h-8 rounded-full bg-white/20 flex items-center justify-center font-bold flex-shrink-0">3</div>
                      <div>
                        <p className="font-semibold mb-1">Get Results</p>
                        <p className="text-sm text-white/90">Review detailed classification with confidence scores.</p>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-xl border border-gray-200/50 p-6">
                  <h3 className="font-bold text-lg text-gray-900 mb-4">Supported Classifications</h3>
                  <div className="grid grid-cols-2 gap-3">
                    {[
                      { name: 'Melanoma', color: '#8b5cf6' },
                      { name: 'Basal Cell', color: '#ef4444' },
                      { name: 'Actinic Keratosis', color: '#f97316' },
                      { name: 'Benign Keratosis', color: '#10b981' },
                      { name: 'Dermatofibroma', color: '#3b82f6' },
                      { name: 'Melanocytic Nevus', color: '#06b6d4' },
                      { name: 'Vascular Lesion', color: '#ec4899' }
                    ].map((item, idx) => (
                      <div key={idx} className="flex items-center gap-2 p-2 bg-gray-50 rounded-lg">
                        <div className="w-3 h-3 rounded-full" style={{ backgroundColor: item.color }}></div>
                        <span className="text-xs font-medium text-gray-700">{item.name}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      </main>

      <footer className="mt-12 py-6 border-t border-gray-200/50 bg-white/30 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-4 text-center">
          <p className="text-sm font-semibold text-gray-700">
            SkinFusion-Net v2.0 with Validated Detection
          </p>
          <p className="text-xs text-gray-600 mt-1">
            Powered by ConvNeXt • EfficientNet • ResNet Ensemble
          </p>
        </div>
      </footer>
    </div>
  );
}
