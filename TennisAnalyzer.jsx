import React, { useState, useEffect, useRef, useCallback } from 'react';

/**
 * Tennis Analysis Visualizer
 * 
 * A React component for visualizing tennis match analysis.
 * Works standalone with simulated data, or can load analysis JSON from the Python pipeline.
 * 
 * Based on:
 * - CourtCheck (AggieSportsAnalytics)
 * - yastrebksv/TennisProject
 * - roboflow/sports
 */

// Court keypoint configuration (17 points as per CourtCheck)
const KEYPOINT_CONFIG = {
  names: [
    'BTL', 'BTLI', 'BTRI', 'BTR',  // Top baseline
    'ITL', 'ITM', 'ITR',            // Service line top
    'NL', 'NM', 'NR',               // Net
    'IBL', 'IBM', 'IBR',            // Service line bottom
    'BBL', 'BBLI', 'BBRI', 'BBR'    // Bottom baseline
  ],
  
  lines: [
    // Baselines
    ['BTL', 'BTLI'], ['BTLI', 'BTRI'], ['BTRI', 'BTR'],
    ['BBL', 'BBLI'], ['BBLI', 'BBRI'], ['BBRI', 'BBR'],
    // Sidelines
    ['BTL', 'NL'], ['NL', 'BBL'],
    ['BTR', 'NR'], ['NR', 'BBR'],
    // Singles lines  
    ['BTLI', 'ITL'], ['ITL', 'IBL'], ['IBL', 'BBLI'],
    ['BTRI', 'ITR'], ['ITR', 'IBR'], ['IBR', 'BBRI'],
    // Service lines
    ['ITL', 'ITM'], ['ITM', 'ITR'],
    ['IBL', 'IBM'], ['IBM', 'IBR'],
    // Center service
    ['ITM', 'NM'], ['NM', 'IBM'],
    // Net
    ['NL', 'NM'], ['NM', 'NR']
  ]
};

// Generate reference court keypoints
const generateCourtKeypoints = (width, height, margin = 20) => {
  const w = width - 2 * margin;
  const h = height - 2 * margin;
  
  const singlesOffset = w * 0.125;  // ~12.5% for doubles alley
  const serviceY = h * 0.27;        // Service line distance
  
  return {
    BTL: [margin, margin],
    BTLI: [margin + singlesOffset, margin],
    BTRI: [margin + w - singlesOffset, margin],
    BTR: [margin + w, margin],
    
    ITL: [margin + singlesOffset, margin + serviceY],
    ITM: [margin + w/2, margin + serviceY],
    ITR: [margin + w - singlesOffset, margin + serviceY],
    
    NL: [margin, margin + h/2],
    NM: [margin + w/2, margin + h/2],
    NR: [margin + w, margin + h/2],
    
    IBL: [margin + singlesOffset, margin + h - serviceY],
    IBM: [margin + w/2, margin + h - serviceY],
    IBR: [margin + w - singlesOffset, margin + h - serviceY],
    
    BBL: [margin, margin + h],
    BBLI: [margin + singlesOffset, margin + h],
    BBRI: [margin + w - singlesOffset, margin + h],
    BBR: [margin + w, margin + h]
  };
};

// Ball tracker class
class BallTracker {
  constructor(maxHistory = 30) {
    this.history = [];
    this.maxHistory = maxHistory;
    this.velocity = { x: 0, y: 0 };
  }
  
  update(pos) {
    if (this.history.length > 0) {
      const last = this.history[this.history.length - 1];
      this.velocity = { x: pos.x - last.x, y: pos.y - last.y };
    }
    this.history.push({ ...pos, t: Date.now() });
    if (this.history.length > this.maxHistory) this.history.shift();
  }
  
  getTrail(length = 15) {
    return this.history.slice(-length);
  }
  
  getSpeed() {
    return Math.sqrt(this.velocity.x ** 2 + this.velocity.y ** 2);
  }
  
  predict() {
    if (this.history.length === 0) return null;
    const last = this.history[this.history.length - 1];
    return {
      x: last.x + this.velocity.x,
      y: last.y + this.velocity.y
    };
  }
}

// Bounce detector
class BounceDetector {
  constructor(threshold = 8, minFramesBetween = 20) {
    this.positions = [];
    this.threshold = threshold;
    this.minFramesBetween = minFramesBetween;
    this.bounces = [];
    this.lastBounceFrame = -100;
  }
  
  update(pos, frame) {
    this.positions.push({ ...pos, frame });
    if (this.positions.length > 10) this.positions.shift();
    
    if (this.positions.length < 5) return null;
    if (frame - this.lastBounceFrame < this.minFramesBetween) return null;
    
    const recent = this.positions.slice(-5);
    const velocities = [];
    for (let i = 1; i < recent.length; i++) {
      velocities.push(recent[i].y - recent[i-1].y);
    }
    
    for (let i = 1; i < velocities.length; i++) {
      if (velocities[i-1] > this.threshold && velocities[i] < -this.threshold) {
        const bounce = {
          position: { x: recent[i].x, y: recent[i].y },
          frame,
          isIn: this.checkInBounds(recent[i])
        };
        this.bounces.push(bounce);
        this.lastBounceFrame = frame;
        return bounce;
      }
    }
    return null;
  }
  
  checkInBounds(pos) {
    // Simplified bounds check
    return pos.x > 100 && pos.x < 540 && pos.y > 60 && pos.y < 420;
  }
  
  getRecent(count = 5) {
    return this.bounces.slice(-count);
  }
}

// Homography transformer (simplified)
const transformPoint = (point, srcCorners, dstCorners) => {
  // Simple linear interpolation for demo
  // Real implementation would use cv2.perspectiveTransform
  const srcBounds = {
    minX: Math.min(...srcCorners.map(p => p[0])),
    maxX: Math.max(...srcCorners.map(p => p[0])),
    minY: Math.min(...srcCorners.map(p => p[1])),
    maxY: Math.max(...srcCorners.map(p => p[1]))
  };
  const dstBounds = {
    minX: Math.min(...dstCorners.map(p => p[0])),
    maxX: Math.max(...dstCorners.map(p => p[0])),
    minY: Math.min(...dstCorners.map(p => p[1])),
    maxY: Math.max(...dstCorners.map(p => p[1]))
  };
  
  const normX = (point.x - srcBounds.minX) / (srcBounds.maxX - srcBounds.minX);
  const normY = (point.y - srcBounds.minY) / (srcBounds.maxY - srcBounds.minY);
  
  return {
    x: dstBounds.minX + normX * (dstBounds.maxX - dstBounds.minX),
    y: dstBounds.minY + normY * (dstBounds.maxY - dstBounds.minY)
  };
};

// Main component
export default function TennisAnalyzer() {
  // State
  const [isPlaying, setIsPlaying] = useState(false);
  const [frame, setFrame] = useState(0);
  const [courtSurface, setCourtSurface] = useState('hard');
  const [showKeypoints, setShowKeypoints] = useState(true);
  const [showTrajectory, setShowTrajectory] = useState(true);
  const [showBounces, setShowBounces] = useState(true);
  const [show2DCourt, setShow2DCourt] = useState(true);
  
  // Refs
  const mainCanvasRef = useRef(null);
  const courtCanvasRef = useRef(null);
  const ballTrackerRef = useRef(new BallTracker());
  const bounceDetectorRef = useRef(new BounceDetector());
  
  // Dimensions
  const MAIN = { width: 640, height: 480 };
  const COURT = { width: 220, height: 440 };
  
  // Court colors
  const surfaceColors = {
    hard: { bg: '#1e40af', lines: '#ffffff' },
    clay: { bg: '#c2410c', lines: '#ffffff' },
    grass: { bg: '#166534', lines: '#ffffff' }
  };
  
  // Players state
  const [players, setPlayers] = useState([
    { id: 1, x: 320, y: 380, color: '#ef4444', speed: 0, distance: 0 },
    { id: 2, x: 320, y: 100, color: '#3b82f6', speed: 0, distance: 0 }
  ]);
  
  // Ball state
  const [ball, setBall] = useState({ x: 320, y: 240, speed: 0 });
  
  // Bounces
  const [bounces, setBounces] = useState([]);
  
  // Keypoints (camera view)
  const cameraKeypoints = generateCourtKeypoints(MAIN.width, MAIN.height, 80);
  cameraKeypoints.BTL = [80, 50];
  cameraKeypoints.BTR = [560, 50];
  cameraKeypoints.BBL = [80, 430];
  cameraKeypoints.BBR = [560, 430];
  
  // 2D court keypoints
  const court2DKeypoints = generateCourtKeypoints(COURT.width, COURT.height, 15);
  
  // Simulation update
  const updateSimulation = useCallback(() => {
    const t = frame * 0.03;
    const tracker = ballTrackerRef.current;
    const bounceDetector = bounceDetectorRef.current;
    
    // Update ball
    const rallyPhase = (t % 4) / 4;
    const goingDown = rallyPhase < 0.5;
    const phase = goingDown ? rallyPhase * 2 : 2 - rallyPhase * 2;
    
    const newBallX = 320 + Math.sin(t * 2.5) * 180;
    const newBallY = 80 + phase * 320;
    
    tracker.update({ x: newBallX, y: newBallY });
    const speed = tracker.getSpeed() * 40;
    setBall({ x: newBallX, y: newBallY, speed });
    
    // Check for bounce
    const bounce = bounceDetector.update({ x: newBallX, y: newBallY }, frame);
    if (bounce) {
      setBounces(prev => [...prev.slice(-10), bounce]);
    }
    
    // Update players
    setPlayers(prev => prev.map((p, i) => {
      const baseY = i === 0 ? 380 : 100;
      const newX = 320 + Math.sin(t * (1 + i * 0.3)) * 150;
      const newY = baseY + Math.cos(t * (0.7 + i * 0.2)) * 40;
      const dist = Math.hypot(newX - p.x, newY - p.y);
      return {
        ...p,
        x: newX,
        y: newY,
        speed: dist * 30,
        distance: p.distance + dist * 0.01
      };
    }));
    
    setFrame(f => f + 1);
  }, [frame]);
  
  // Animation loop
  useEffect(() => {
    if (!isPlaying) return;
    const interval = setInterval(updateSimulation, 33);
    return () => clearInterval(interval);
  }, [isPlaying, updateSimulation]);
  
  // Draw main view
  useEffect(() => {
    const canvas = mainCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const colors = surfaceColors[courtSurface];
    
    // Clear
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, MAIN.width, MAIN.height);
    
    // Court surface
    ctx.fillStyle = colors.bg;
    ctx.beginPath();
    ctx.moveTo(cameraKeypoints.BTL[0], cameraKeypoints.BTL[1]);
    ctx.lineTo(cameraKeypoints.BTR[0], cameraKeypoints.BTR[1]);
    ctx.lineTo(cameraKeypoints.BBR[0], cameraKeypoints.BBR[1]);
    ctx.lineTo(cameraKeypoints.BBL[0], cameraKeypoints.BBL[1]);
    ctx.closePath();
    ctx.fill();
    
    // Court lines
    ctx.strokeStyle = colors.lines;
    ctx.lineWidth = 2;
    
    KEYPOINT_CONFIG.lines.forEach(([start, end]) => {
      const p1 = cameraKeypoints[start];
      const p2 = cameraKeypoints[end];
      if (p1 && p2) {
        ctx.beginPath();
        ctx.moveTo(p1[0], p1[1]);
        ctx.lineTo(p2[0], p2[1]);
        ctx.stroke();
      }
    });
    
    // Net (thicker)
    ctx.lineWidth = 4;
    ctx.strokeStyle = '#f8fafc';
    ctx.beginPath();
    ctx.moveTo(cameraKeypoints.NL[0], cameraKeypoints.NL[1]);
    ctx.lineTo(cameraKeypoints.NR[0], cameraKeypoints.NR[1]);
    ctx.stroke();
    
    // Keypoints
    if (showKeypoints) {
      Object.entries(cameraKeypoints).forEach(([name, pos]) => {
        ctx.beginPath();
        ctx.fillStyle = '#22c55e';
        ctx.arc(pos[0], pos[1], 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1;
        ctx.stroke();
      });
    }
    
    // Ball trajectory
    if (showTrajectory) {
      const trail = ballTrackerRef.current.getTrail();
      trail.forEach((pos, i) => {
        const alpha = (i + 1) / trail.length;
        ctx.beginPath();
        ctx.fillStyle = `rgba(250, 204, 21, ${alpha * 0.7})`;
        ctx.arc(pos.x, pos.y, 2 + alpha * 3, 0, Math.PI * 2);
        ctx.fill();
      });
    }
    
    // Bounces
    if (showBounces) {
      bounces.forEach(b => {
        ctx.beginPath();
        ctx.strokeStyle = b.isIn ? '#22c55e' : '#ef4444';
        ctx.lineWidth = 3;
        ctx.arc(b.position.x, b.position.y, 12, 0, Math.PI * 2);
        ctx.stroke();
        
        ctx.fillStyle = b.isIn ? '#22c55e' : '#ef4444';
        ctx.font = 'bold 10px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(b.isIn ? 'IN' : 'OUT', b.position.x, b.position.y + 25);
      });
    }
    
    // Ball
    const ballGrad = ctx.createRadialGradient(ball.x - 2, ball.y - 2, 0, ball.x, ball.y, 8);
    ballGrad.addColorStop(0, '#fef08a');
    ballGrad.addColorStop(0.6, '#facc15');
    ballGrad.addColorStop(1, '#ca8a04');
    ctx.fillStyle = ballGrad;
    ctx.beginPath();
    ctx.arc(ball.x, ball.y, 7, 0, Math.PI * 2);
    ctx.fill();
    
    ctx.fillStyle = '#facc15';
    ctx.font = 'bold 11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(`${ball.speed.toFixed(0)} km/h`, ball.x, ball.y - 15);
    
    // Players
    players.forEach(p => {
      ctx.beginPath();
      ctx.fillStyle = p.color;
      ctx.arc(p.x, p.y, 12, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.stroke();
      
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 10px sans-serif';
      ctx.fillText(`P${p.id}`, p.x, p.y + 4);
    });
    
    // Frame counter
    ctx.fillStyle = '#fff';
    ctx.font = '12px monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`Frame: ${frame}`, 10, 20);
    ctx.fillText(`Bounces: ${bounces.length}`, 10, 38);
    
  }, [ball, players, bounces, frame, showKeypoints, showTrajectory, showBounces, courtSurface, cameraKeypoints]);
  
  // Draw 2D court
  useEffect(() => {
    const canvas = courtCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const colors = surfaceColors[courtSurface];
    
    // Background
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, COURT.width, COURT.height);
    
    // Court surface
    ctx.fillStyle = colors.bg;
    ctx.fillRect(15, 15, COURT.width - 30, COURT.height - 30);
    
    // Lines
    ctx.strokeStyle = colors.lines;
    ctx.lineWidth = 1;
    
    KEYPOINT_CONFIG.lines.forEach(([start, end]) => {
      const p1 = court2DKeypoints[start];
      const p2 = court2DKeypoints[end];
      if (p1 && p2) {
        ctx.beginPath();
        ctx.moveTo(p1[0], p1[1]);
        ctx.lineTo(p2[0], p2[1]);
        ctx.stroke();
      }
    });
    
    // Transform and draw players
    const srcCorners = [
      cameraKeypoints.BTL, cameraKeypoints.BTR,
      cameraKeypoints.BBL, cameraKeypoints.BBR
    ];
    const dstCorners = [
      court2DKeypoints.BTL, court2DKeypoints.BTR,
      court2DKeypoints.BBL, court2DKeypoints.BBR
    ];
    
    players.forEach(p => {
      const pos2D = transformPoint({ x: p.x, y: p.y }, srcCorners, dstCorners);
      ctx.beginPath();
      ctx.fillStyle = p.color;
      ctx.arc(pos2D.x, pos2D.y, 6, 0, Math.PI * 2);
      ctx.fill();
    });
    
    // Transform ball
    const ball2D = transformPoint(ball, srcCorners, dstCorners);
    ctx.beginPath();
    ctx.fillStyle = '#facc15';
    ctx.arc(ball2D.x, ball2D.y, 5, 0, Math.PI * 2);
    ctx.fill();
    
    // Transform bounces
    if (showBounces) {
      bounces.slice(-5).forEach(b => {
        const pos2D = transformPoint(b.position, srcCorners, dstCorners);
        ctx.beginPath();
        ctx.strokeStyle = b.isIn ? '#22c55e' : '#ef4444';
        ctx.lineWidth = 2;
        ctx.arc(pos2D.x, pos2D.y, 6, 0, Math.PI * 2);
        ctx.stroke();
      });
    }
    
  }, [ball, players, bounces, showBounces, courtSurface, court2DKeypoints, cameraKeypoints]);
  
  // Stats panel
  const StatsPanel = ({ player }) => (
    <div className="bg-gray-800/60 rounded p-2 border-l-2" style={{ borderColor: player.color }}>
      <div className="text-xs text-gray-400">Player {player.id}</div>
      <div className="grid grid-cols-2 gap-1 mt-1 text-xs">
        <div className="bg-gray-900/50 p-1 rounded">
          <span className="text-gray-500">Speed</span>
          <div className="text-white font-mono">{player.speed.toFixed(1)}</div>
        </div>
        <div className="bg-gray-900/50 p-1 rounded">
          <span className="text-gray-500">Dist</span>
          <div className="text-white font-mono">{player.distance.toFixed(1)}m</div>
        </div>
      </div>
    </div>
  );
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-gray-900 to-slate-900 text-white p-4">
      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <div className="text-center mb-4">
          <h1 className="text-2xl font-bold bg-gradient-to-r from-yellow-400 via-green-400 to-blue-400 bg-clip-text text-transparent">
            🎾 Tennis Match Analyzer
          </h1>
          <p className="text-gray-400 text-xs mt-1">
            Court Detection • Ball Tracking • Bounce Detection • Homography Transform
          </p>
        </div>
        
        {/* Main content */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-3">
          {/* Main view */}
          <div className="lg:col-span-3">
            <div className="bg-gray-800/40 rounded-lg p-3">
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm text-gray-300">Camera View</span>
                <select
                  value={courtSurface}
                  onChange={(e) => setCourtSurface(e.target.value)}
                  className="bg-gray-700 text-xs rounded px-2 py-1"
                >
                  <option value="hard">Hard</option>
                  <option value="clay">Clay</option>
                  <option value="grass">Grass</option>
                </select>
              </div>
              <canvas
                ref={mainCanvasRef}
                width={MAIN.width}
                height={MAIN.height}
                className="w-full rounded"
              />
            </div>
            
            {/* Controls */}
            <div className="bg-gray-800/40 rounded-lg p-3 mt-2">
              <div className="flex flex-wrap items-center gap-2">
                <button
                  onClick={() => setIsPlaying(!isPlaying)}
                  className={`px-4 py-1.5 rounded text-sm font-medium ${
                    isPlaying ? 'bg-red-500' : 'bg-green-500'
                  }`}
                >
                  {isPlaying ? '⏸ Pause' : '▶ Play'}
                </button>
                <button
                  onClick={() => {
                    setFrame(0);
                    setBounces([]);
                    ballTrackerRef.current = new BallTracker();
                    bounceDetectorRef.current = new BounceDetector();
                  }}
                  className="px-4 py-1.5 rounded bg-gray-600 text-sm"
                >
                  ↺ Reset
                </button>
                
                <div className="flex gap-2 ml-auto">
                  {[
                    { label: 'Keypoints', state: showKeypoints, set: setShowKeypoints },
                    { label: 'Trajectory', state: showTrajectory, set: setShowTrajectory },
                    { label: 'Bounces', state: showBounces, set: setShowBounces },
                    { label: '2D Court', state: show2DCourt, set: setShow2DCourt },
                  ].map(opt => (
                    <button
                      key={opt.label}
                      onClick={() => opt.set(!opt.state)}
                      className={`px-2 py-1 rounded text-xs ${
                        opt.state ? 'bg-blue-500' : 'bg-gray-700'
                      }`}
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>
          
          {/* Side panel */}
          <div className="space-y-2">
            {/* 2D Court */}
            {show2DCourt && (
              <div className="bg-gray-800/40 rounded-lg p-2">
                <div className="text-xs text-gray-400 mb-1">2D Court View</div>
                <canvas
                  ref={courtCanvasRef}
                  width={COURT.width}
                  height={COURT.height}
                  className="w-full rounded"
                />
              </div>
            )}
            
            {/* Ball stats */}
            <div className="bg-gray-800/40 rounded-lg p-2">
              <div className="flex items-center gap-1 text-xs text-gray-400 mb-1">
                <div className="w-2 h-2 rounded-full bg-yellow-400" />
                Ball Tracking
              </div>
              <div className="grid grid-cols-2 gap-1 text-xs">
                <div className="bg-gray-900/50 p-1.5 rounded text-center">
                  <div className="text-gray-500">Speed</div>
                  <div className="text-yellow-400 font-mono">{ball.speed.toFixed(0)} km/h</div>
                </div>
                <div className="bg-gray-900/50 p-1.5 rounded text-center">
                  <div className="text-gray-500">Bounces</div>
                  <div className="text-green-400 font-mono">{bounces.length}</div>
                </div>
              </div>
            </div>
            
            {/* Player stats */}
            {players.map(p => <StatsPanel key={p.id} player={p} />)}
          </div>
        </div>
        
        {/* Footer */}
        <div className="text-center text-gray-500 text-xs mt-4">
          Based on CourtCheck, TrackNet, and roboflow/sports
        </div>
      </div>
    </div>
  );
}
