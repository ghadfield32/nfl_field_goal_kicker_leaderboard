import React, { useState, useEffect, useMemo } from 'react';
import { Upload, TrendingUp, Target, BarChart3, Users, Play, Download, Filter, Search, ArrowUpDown } from 'lucide-react';

// Mock data based on your document - this would be replaced with actual CSV data
const mockKickerData = [
  { player_id: 46298, player_name: "Jason Sanders", team: "MIA", rating: 0.4715, rank: 1, attempts: 12, made: 12, confidence_interval: [0.15, 0.85] },
  { player_id: 45678, player_name: "Brett Maher", team: "DAL", rating: 0.30, rank: 2, attempts: 15, made: 14, confidence_interval: [0.10, 0.55] },
  { player_id: 23456, player_name: "Justin Tucker", team: "BAL", rating: 0.22, rank: 3, attempts: 45, made: 42, confidence_interval: [0.18, 0.26] },
  { player_id: 34567, player_name: "Harrison Butker", team: "KC", rating: 0.20, rank: 4, attempts: 38, made: 35, confidence_interval: [0.15, 0.25] },
  { player_id: 12345, player_name: "Matt Bryant", team: "ATL", rating: 0.16, rank: 5, attempts: 42, made: 38, confidence_interval: [0.12, 0.20] },
  { player_id: 56789, player_name: "Dan Bailey", team: "MIN", rating: 0.13, rank: 6, attempts: 35, made: 32, confidence_interval: [0.08, 0.18] },
  { player_id: 67890, player_name: "Adam Vinatieri", team: "IND", rating: 0.13, rank: 7, attempts: 28, made: 26, confidence_interval: [0.07, 0.19] },
  { player_id: 78901, player_name: "Stephen Gostkowski", team: "NE", rating: 0.12, rank: 10, attempts: 32, made: 29, confidence_interval: [0.06, 0.18] },
  { player_id: 89012, player_name: "Sam Ficken", team: "NYJ", rating: -1.57, rank: 86, attempts: 6, made: 1, confidence_interval: [-2.5, -0.8] },
  { player_id: 90123, player_name: "Nick Folk", team: "NE", rating: -0.50, rank: 84, attempts: 18, made: 12, confidence_interval: [-0.8, -0.2] }
];

const mockAttemptData = [
  { distance: 25, made: 1, kicker: "Jason Sanders" },
  { distance: 35, made: 1, kicker: "Jason Sanders" },
  { distance: 42, made: 1, kicker: "Justin Tucker" },
  { distance: 58, made: 1, kicker: "Justin Tucker" },
  { distance: 45, made: 0, kicker: "Sam Ficken" },
  { distance: 38, made: 0, kicker: "Sam Ficken" }
];

const NFLKickerAnalytics = () => {
  const [activeTab, setActiveTab] = useState('leaderboard');
  const [kickerData, setKickerData] = useState(mockKickerData);
  const [attemptData, setAttemptData] = useState(mockAttemptData);
  const [searchTerm, setSearchTerm] = useState('');
  const [sortField, setSortField] = useState('rank');
  const [sortDirection, setSortDirection] = useState('asc');
  const [filterTeam, setFilterTeam] = useState('');
  const [simulationRuns, setSimulationRuns] = useState(1000);
  const [selectedKicker, setSelectedKicker] = useState(null);
  const [simulationDistance, setSimulationDistance] = useState(45);
  const [simulationResults, setSimulationResults] = useState(null);

  // Get unique teams for filter
  const teams = useMemo(() => {
    const teamSet = new Set(kickerData.map(k => k.team));
    return Array.from(teamSet).sort();
  }, [kickerData]);

  // Filter and sort kicker data
  const filteredKickers = useMemo(() => {
    let filtered = kickerData.filter(kicker => {
      const matchesSearch = kicker.player_name.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesTeam = !filterTeam || kicker.team === filterTeam;
      return matchesSearch && matchesTeam;
    });

    return filtered.sort((a, b) => {
      const aVal = a[sortField];
      const bVal = b[sortField];
      const multiplier = sortDirection === 'asc' ? 1 : -1;
      
      if (typeof aVal === 'string') {
        return aVal.localeCompare(bVal) * multiplier;
      }
      return (aVal - bVal) * multiplier;
    });
  }, [kickerData, searchTerm, filterTeam, sortField, sortDirection]);

  // Handle CSV file upload
  const handleFileUpload = (event, dataType) => {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      const csv = e.target.result;
      const lines = csv.split('\n');
      const headers = lines[0].split(',').map(h => h.trim());
      
      const data = lines.slice(1).filter(line => line.trim()).map(line => {
        const values = line.split(',');
        const obj = {};
        headers.forEach((header, index) => {
          obj[header] = values[index]?.trim();
        });
        return obj;
      });

      if (dataType === 'kickers') {
        // Process kicker data upload
        console.log('Uploaded kicker data:', data);
        // Here you would merge with existing data and recalculate ratings
      } else if (dataType === 'attempts') {
        // Process attempt data upload
        console.log('Uploaded attempt data:', data);
        setAttemptData(data);
      }
    };
    reader.readAsText(file);
  };

  // Simulate field goal success for a kicker at a specific distance
  const simulateFieldGoal = (kicker, distance, runs = simulationRuns) => {
    // Bayesian model simulation (simplified)
    // In reality, this would use the full hierarchical model parameters
    const baseSuccessRate = Math.max(0.1, Math.min(0.99, 1.0 - (distance - 18) * 0.015));
    const kickerAdjustment = kicker.rating * 0.5; // Simplified adjustment
    const adjustedRate = Math.max(0.05, Math.min(0.95, baseSuccessRate + kickerAdjustment));
    
    let successes = 0;
    for (let i = 0; i < runs; i++) {
      if (Math.random() < adjustedRate) successes++;
    }
    
    return {
      successRate: successes / runs,
      expectedPoints: (successes / runs) * 3,
      confidenceInterval: [
        Math.max(0, (successes / runs) - 1.96 * Math.sqrt((successes / runs) * (1 - successes / runs) / runs)),
        Math.min(1, (successes / runs) + 1.96 * Math.sqrt((successes / runs) * (1 - successes / runs) / runs))
      ]
    };
  };

  // Run simulation for selected kicker and distance
  const runSimulation = () => {
    if (!selectedKicker) return;
    
    const kicker = kickerData.find(k => k.player_id === selectedKicker);
    const result = simulateFieldGoal(kicker, simulationDistance, simulationRuns);
    setSimulationResults({
      kicker: kicker.player_name,
      distance: simulationDistance,
      runs: simulationRuns,
      ...result
    });
  };

  const handleSort = (field) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('asc');
    }
  };

  const getRatingColor = (rating) => {
    if (rating > 0.2) return 'text-green-600 font-bold';
    if (rating > 0) return 'text-green-500';
    if (rating > -0.2) return 'text-yellow-600';
    return 'text-red-600 font-bold';
  };

  const getConfidenceWidth = (interval) => {
    return (interval[1] - interval[0]) * 100;
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div className="flex items-center">
              <Target className="h-8 w-8 text-blue-600 mr-3" />
              <h1 className="text-3xl font-bold text-gray-900">NFL Kicker Analytics</h1>
            </div>
            <div className="text-sm text-gray-500">
              Bayesian Rating System • Week 6, 2018
            </div>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <div className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <nav className="flex space-x-8 py-4">
            {[
              { id: 'leaderboard', label: 'Leaderboard', icon: BarChart3 },
              { id: 'simulator', label: 'Simulator', icon: Play },
              { id: 'insights', label: 'Model Insights', icon: TrendingUp },
              { id: 'data', label: 'Data Management', icon: Upload }
            ].map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center px-3 py-2 text-sm font-medium rounded-md ${
                    activeTab === tab.id
                      ? 'bg-blue-100 text-blue-700'
                      : 'text-gray-500 hover:text-gray-700'
                  }`}
                >
                  <Icon className="h-4 w-4 mr-2" />
                  {tab.label}
                </button>
              );
            })}
          </nav>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'leaderboard' && (
          <div className="space-y-6">
            {/* Controls */}
            <div className="bg-white rounded-lg shadow-sm p-6">
              <div className="flex flex-wrap gap-4 items-center justify-between">
                <div className="flex items-center space-x-4">
                  <div className="relative">
                    <Search className="h-4 w-4 absolute left-3 top-3 text-gray-400" />
                    <input
                      type="text"
                      placeholder="Search kickers..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>
                  <select
                    value={filterTeam}
                    onChange={(e) => setFilterTeam(e.target.value)}
                    className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="">All Teams</option>
                    {teams.map(team => (
                      <option key={team} value={team}>{team}</option>
                    ))}
                  </select>
                </div>
                <div className="flex items-center space-x-2">
                  <Filter className="h-4 w-4 text-gray-400" />
                  <span className="text-sm text-gray-500">
                    Showing {filteredKickers.length} kickers
                  </span>
                </div>
              </div>
            </div>

            {/* Leaderboard Table */}
            <div className="bg-white rounded-lg shadow-sm overflow-hidden">
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      {[
                        { key: 'rank', label: 'Rank' },
                        { key: 'player_name', label: 'Player' },
                        { key: 'team', label: 'Team' },
                        { key: 'rating', label: 'EPA-FG+' },
                        { key: 'attempts', label: 'Attempts' },
                        { key: 'made', label: 'Made' }
                      ].map((column) => (
                        <th
                          key={column.key}
                          onClick={() => handleSort(column.key)}
                          className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                        >
                          <div className="flex items-center space-x-1">
                            <span>{column.label}</span>
                            <ArrowUpDown className="h-3 w-3" />
                          </div>
                        </th>
                      ))}
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Confidence
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {filteredKickers.map((kicker) => (
                      <tr key={kicker.player_id} className="hover:bg-gray-50">
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="flex items-center">
                            <span className={`inline-flex items-center justify-center w-8 h-8 rounded-full text-sm font-medium ${
                              kicker.rank <= 3 ? 'bg-yellow-100 text-yellow-800' :
                              kicker.rank <= 10 ? 'bg-green-100 text-green-800' :
                              'bg-gray-100 text-gray-800'
                            }`}>
                              {kicker.rank}
                            </span>
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="text-sm font-medium text-gray-900">
                            {kicker.player_name}
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                            {kicker.team}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className={`text-sm ${getRatingColor(kicker.rating)}`}>
                            {kicker.rating > 0 ? '+' : ''}{kicker.rating.toFixed(3)}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {kicker.attempts}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {kicker.made} ({((kicker.made / kicker.attempts) * 100).toFixed(1)}%)
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="flex items-center">
                            <div className="w-16 bg-gray-200 rounded-full h-2">
                              <div
                                className={`h-2 rounded-full ${
                                  getConfidenceWidth(kicker.confidence_interval) < 20 ? 'bg-green-400' :
                                  getConfidenceWidth(kicker.confidence_interval) < 40 ? 'bg-yellow-400' :
                                  'bg-red-400'
                                }`}
                                style={{
                                  width: `${Math.min(100, getConfidenceWidth(kicker.confidence_interval) * 2)}%`
                                }}
                              />
                            </div>
                            <span className="ml-2 text-xs text-gray-500">
                              ±{(getConfidenceWidth(kicker.confidence_interval) / 2).toFixed(2)}
                            </span>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'simulator' && (
          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow-sm p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-6">Field Goal Simulation</h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Select Kicker
                  </label>
                  <select
                    value={selectedKicker || ''}
                    onChange={(e) => setSelectedKicker(parseInt(e.target.value))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="">Choose a kicker...</option>
                    {kickerData.map(kicker => (
                      <option key={kicker.player_id} value={kicker.player_id}>
                        {kicker.player_name} ({kicker.team})
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Distance (yards)
                  </label>
                  <input
                    type="range"
                    min="18"
                    max="65"
                    value={simulationDistance}
                    onChange={(e) => setSimulationDistance(parseInt(e.target.value))}
                    className="w-full"
                  />
                  <div className="text-center text-sm text-gray-600 mt-1">
                    {simulationDistance} yards
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Simulation Runs
                  </label>
                  <select
                    value={simulationRuns}
                    onChange={(e) => setSimulationRuns(parseInt(e.target.value))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value={100}>100 runs</option>
                    <option value={1000}>1,000 runs</option>
                    <option value={10000}>10,000 runs</option>
                    <option value={50000}>50,000 runs</option>
                  </select>
                </div>

                <div className="flex items-end">
                  <button
                    onClick={runSimulation}
                    disabled={!selectedKicker}
                    className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-medium py-2 px-4 rounded-lg flex items-center justify-center"
                  >
                    <Play className="h-4 w-4 mr-2" />
                    Run Simulation
                  </button>
                </div>
              </div>

              {simulationResults && (
                <div className="bg-gray-50 rounded-lg p-6">
                  <h3 className="text-lg font-medium text-gray-900 mb-4">
                    Simulation Results: {simulationResults.kicker}
                  </h3>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                    <div className="bg-white rounded-lg p-4">
                      <div className="text-2xl font-bold text-blue-600">
                        {(simulationResults.successRate * 100).toFixed(1)}%
                      </div>
                      <div className="text-sm text-gray-600">Success Rate</div>
                      <div className="text-xs text-gray-500 mt-1">
                        95% CI: [{(simulationResults.confidenceInterval[0] * 100).toFixed(1)}%, {(simulationResults.confidenceInterval[1] * 100).toFixed(1)}%]
                      </div>
                    </div>
                    <div className="bg-white rounded-lg p-4">
                      <div className="text-2xl font-bold text-green-600">
                        {simulationResults.expectedPoints.toFixed(2)}
                      </div>
                      <div className="text-sm text-gray-600">Expected Points</div>
                      <div className="text-xs text-gray-500 mt-1">
                        From {simulationResults.distance} yards
                      </div>
                    </div>
                    <div className="bg-white rounded-lg p-4">
                      <div className="text-2xl font-bold text-purple-600">
                        {simulationResults.runs.toLocaleString()}
                      </div>
                      <div className="text-sm text-gray-600">Simulations</div>
                      <div className="text-xs text-gray-500 mt-1">
                        Monte Carlo runs
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'data' && (
          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow-sm p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-6">Data Management</h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="border-2 border-dashed border-gray-300 rounded-lg p-6">
                  <div className="text-center">
                    <Upload className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                    <h3 className="text-lg font-medium text-gray-900 mb-2">
                      Upload Field Goal Attempts
                    </h3>
                    <p className="text-sm text-gray-600 mb-4">
                      CSV file with attempt data (player_id, distance, result, etc.)
                    </p>
                    <input
                      type="file"
                      accept=".csv"
                      onChange={(e) => handleFileUpload(e, 'attempts')}
                      className="hidden"
                      id="attempts-upload"
                    />
                    <label
                      htmlFor="attempts-upload"
                      className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-lg cursor-pointer inline-flex items-center"
                    >
                      <Upload className="h-4 w-4 mr-2" />
                      Choose File
                    </label>
                  </div>
                </div>

                <div className="border-2 border-dashed border-gray-300 rounded-lg p-6">
                  <div className="text-center">
                    <Users className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                    <h3 className="text-lg font-medium text-gray-900 mb-2">
                      Upload Kicker Metadata
                    </h3>
                    <p className="text-sm text-gray-600 mb-4">
                      CSV file with kicker information (player_id, name, team, etc.)
                    </p>
                    <input
                      type="file"
                      accept=".csv"
                      onChange={(e) => handleFileUpload(e, 'kickers')}
                      className="hidden"
                      id="kickers-upload"
                    />
                    <label
                      htmlFor="kickers-upload"
                      className="bg-green-600 hover:bg-green-700 text-white font-medium py-2 px-4 rounded-lg cursor-pointer inline-flex items-center"
                    >
                      <Upload className="h-4 w-4 mr-2" />
                      Choose File
                    </label>
                  </div>
                </div>
              </div>

              <div className="mt-8 pt-6 border-t border-gray-200">
                <h3 className="text-lg font-medium text-gray-900 mb-4">Export Data</h3>
                <div className="flex space-x-4">
                  <button className="bg-gray-600 hover:bg-gray-700 text-white font-medium py-2 px-4 rounded-lg inline-flex items-center">
                    <Download className="h-4 w-4 mr-2" />
                    Export Leaderboard
                  </button>
                  <button className="bg-gray-600 hover:bg-gray-700 text-white font-medium py-2 px-4 rounded-lg inline-flex items-center">
                    <Download className="h-4 w-4 mr-2" />
                    Export Simulation Results
                  </button>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg shadow-sm p-6">
              <h3 className="text-lg font-medium text-gray-900 mb-4">Current Data Status</h3>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div className="bg-blue-50 rounded-lg p-4">
                  <div className="text-2xl font-bold text-blue-600">{kickerData.length}</div>
                  <div className="text-sm text-blue-800">Active Kickers</div>
                </div>
                <div className="bg-green-50 rounded-lg p-4">
                  <div className="text-2xl font-bold text-green-600">{attemptData.length}</div>
                  <div className="text-sm text-green-800">Field Goal Attempts</div>
                </div>
                <div className="bg-purple-50 rounded-lg p-4">
                  <div className="text-2xl font-bold text-purple-600">2018</div>
                  <div className="text-sm text-purple-800">Season (Week 6)</div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'insights' && (
          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow-sm p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-6">Model Insights</h2>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <h3 className="text-lg font-medium text-gray-900">Rating Distribution</h3>
                  <div className="text-sm text-gray-600">
                    EPA-FG+ ratings across all NFL kickers, showing the distribution of skill levels.
                  </div>
                  <div className="bg-gray-50 rounded-lg p-4 text-center text-gray-500">
                    [Distribution Chart - Coming Soon]
                  </div>
                </div>
                <div className="space-y-4">