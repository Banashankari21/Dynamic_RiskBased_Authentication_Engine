import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const RiskDashboard = () => {
  const [riskData, setRiskData] = useState([]);

  useEffect(() => {
    // Fetch risk data from FastAPI
    const fetchData = async () => {
      try {
        const response = await axios.get('http://127.0.0.1:8000/risk/user_001');
        // Wrap as array if single object
        setRiskData([response.data]);
      } catch (error) {
        console.error('Error fetching risk data:', error);
      }
    };

    fetchData();
  }, []);

  return (
    <div style={{ padding: '2rem' }}>
      <h2>User Risk Probability</h2>
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={riskData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="user_id" />
          <YAxis domain={[0,1]} />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="risk_probability" stroke="#8884d8" activeDot={{ r: 8 }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default RiskDashboard;
