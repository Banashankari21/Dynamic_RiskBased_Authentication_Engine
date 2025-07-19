import React, { useState } from 'react';
import axios from 'axios';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, PieChart, Pie, Cell
} from 'recharts';
import { Container, TextField, Button, Typography, Paper } from '@mui/material';

const COLORS = ['#0088FE', '#FF8042'];

function App() {
  const [userId, setUserId] = useState('user_001');
  const [historyData, setHistoryData] = useState([]);
  const [latestRisk, setLatestRisk] = useState(null);

  const handleFetchHistory = async () => {
    try {
      // Fetch history data
      const historyResponse = await axios.get(`http://localhost:8000/risk/history/${userId}`);
      const parsedHistory = historyResponse.data.map((item, index) => ({
        index: index + 1,
        risk_probability: item.risk_probability,
        risk_label: item.risk_label
      }));
      setHistoryData(parsedHistory);

      // Fetch latest risk data
      const latestResponse = await axios.get(`http://localhost:8000/risk/${userId}`);
      setLatestRisk(latestResponse.data);

    } catch (error) {
      console.error(error);
      setHistoryData([]);
      setLatestRisk(null);
      alert("No data found or server error.");
    }
  };

  // Pie chart data preparation
  const riskCounts = [
    { name: 'Non-Risky', value: historyData.filter(d => d.risk_label === 0).length },
    { name: 'Risky', value: historyData.filter(d => d.risk_label === 1).length }
  ];

  return (
    <Container maxWidth="md" style={{ marginTop: "40px" }}>
      <Typography variant="h4" gutterBottom>
        🔒 Dynamic Risk-Based Authentication Dashboard
      </Typography>

      <Paper style={{ padding: "20px", marginBottom: "20px" }}>
        <TextField
          label="User ID"
          variant="outlined"
          value={userId}
          onChange={e => setUserId(e.target.value)}
          style={{ marginRight: "10px" }}
        />
        <Button variant="contained" color="primary" onClick={handleFetchHistory}>
          Fetch Risk Data
        </Button>
      </Paper>

      {latestRisk && (
        <Paper style={{ padding: "20px", marginBottom: "20px" }}>
          <Typography variant="h6" gutterBottom>
            🔎 Latest Risk Details:
          </Typography>
          <pre>{JSON.stringify(latestRisk, null, 2)}</pre>
        </Paper>
      )}

      {historyData.length > 0 && (
        <>
          <Paper style={{ padding: "20px", marginBottom: "20px" }}>
            <Typography variant="h6" gutterBottom>
              📈 Risk Probability Over Time
            </Typography>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={historyData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="index" />
                <YAxis domain={[0, 1]} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="risk_probability" stroke="#1976d2" activeDot={{ r: 8 }} />
              </LineChart>
            </ResponsiveContainer>
          </Paper>

          <Paper style={{ padding: "20px", marginBottom: "20px" }}>
            <Typography variant="h6" gutterBottom>
              🟩 Risk Labels Distribution
            </Typography>
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={historyData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="index" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="risk_label" fill="#2e7d32" />
              </BarChart>
            </ResponsiveContainer>
          </Paper>

          <Paper style={{ padding: "20px" }}>
            <Typography variant="h6" gutterBottom>
              🥧 Risky vs Non-Risky Pie Chart
            </Typography>
            <ResponsiveContainer width="100%" height={400}>
              <PieChart>
                <Pie
                  data={riskCounts}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  outerRadius={150}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {riskCounts.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </Paper>
        </>
      )}
    </Container>
  );
}

export default App;
