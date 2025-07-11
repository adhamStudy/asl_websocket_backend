# websocket_server.py - FLUTTER CAMERA INPUT VERSION
import asyncio
import websockets
import json
import cv2
import mediapipe as mp
import time
import numpy as np
from collections import deque
import threading
import socket
import base64

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Same configuration as your original code
VALID_FIRST_SIGNS = {'00001', '01000', '01100', '00010','00111'}
IGNORED_SIGNS = {'00000', '00100'}

SEQUENCE_ACTIONS = {
    # 2-sign sequences
    ('00001', '01111'): 'Call Dad',
    ('00001', '01110'): 'Call Mom',
    ('01000', '00111'): 'Open Google',
    ('01000', '01111'): 'Open Siri',
    
    # 3-sign sequences
    ('00001', '01000', '01100'): 'Call Police',
    ('01000', '00010', '00001'): 'Send Message',
    ('00111', '00011', '00001'): 'Open Camera',
    ('01100', '01110', '01111'): 'Play Music',
    ('00001', '00010', '01100'): 'Take Screenshot',
    ('01000', '01100', '00001'): 'Open Calculator',
}

def get_local_ip():
    """Get the local IP address of this machine"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"

class SignStabilizer:
    def __init__(self, history_size=3, stability_threshold=0.6):  # Fast settings
        self.history = deque(maxlen=history_size)
        self.stability_threshold = stability_threshold
    
    def add_detection(self, code):
        if code is not None:
            self.history.append(code)
    
    def get_stable_sign(self):
        if len(self.history) < 2:  # Only require 2 consistent detections for speed
            return None
        
        sign_counts = {}
        for sign in self.history:
            sign_counts[sign] = sign_counts.get(sign, 0) + 1
        
        most_frequent = max(sign_counts.items(), key=lambda x: x[1])
        stability_ratio = most_frequent[1] / len(self.history)
        
        if stability_ratio >= self.stability_threshold:
            return most_frequent[0]
        
        return None
    
    def clear(self):
        self.history.clear()

def is_valid_hand(hand_landmarks):
    """Validate if detected landmarks actually represent a hand, not a face"""
    
    # Get key landmarks
    wrist = hand_landmarks.landmark[0]
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]
    
    # Check 1: Hand span ratio (width vs height)
    tips = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
    
    # Calculate bounding box
    min_x = min(landmark.x for landmark in hand_landmarks.landmark)
    max_x = max(landmark.x for landmark in hand_landmarks.landmark)
    min_y = min(landmark.y for landmark in hand_landmarks.landmark)
    max_y = max(landmark.y for landmark in hand_landmarks.landmark)
    
    width = max_x - min_x
    height = max_y - min_y
    
    if width == 0 or height == 0:
        return False
        
    aspect_ratio = width / height
    
    # Real hands typically have aspect ratio between 0.5 and 2.0
    if aspect_ratio < 0.5 or aspect_ratio > 2.0:
        print(f"❌ Invalid hand: Bad aspect ratio {aspect_ratio:.2f}")
        return False
    
    # Check 2: Hand size validation
    hand_size = max(width, height)
    
    # Reasonable hand size in camera frame
    if hand_size < 0.1 or hand_size > 0.8:
        print(f"❌ Invalid hand: Bad size {hand_size:.3f}")
        return False
    
    return True

def get_finger_code_enhanced(hand_landmarks):
    """Enhanced finger detection with adaptive thresholds and multiple methods"""
    tips_ids = [4, 8, 12, 16, 20]
    pip_ids = [3, 6, 10, 14, 18]
    mcp_ids = [2, 5, 9, 13, 17]
    
    states = []

    # THUMB ALWAYS DISABLED
    states.append('0')

    # Enhanced detection for other fingers
    for i in range(1, 5):
        finger_extended = False
        
        tip = hand_landmarks.landmark[tips_ids[i]]
        pip = hand_landmarks.landmark[pip_ids[i]]
        mcp = hand_landmarks.landmark[mcp_ids[i]]
        
        # METHOD 1: Tip vs PIP comparison
        y_diff_pip = pip.y - tip.y
        
        # STRICT thresholds for index and middle (prevent half-open detection)
        # More lenient for ring and pinky (they're naturally harder to extend)
        if i == 1:  # Index finger - VERY STRICT
            threshold_pip = 0.025  # Much higher threshold for full extension
        elif i == 2:  # Middle finger - VERY STRICT  
            threshold_pip = 0.023  # Much higher threshold for full extension
        elif i == 3:  # Ring finger (keep moderate)
            threshold_pip = 0.010
        elif i == 4:  # Pinky finger (keep sensitive)
            threshold_pip = 0.008
        
        method1_extended = y_diff_pip > threshold_pip
        
        # METHOD 2: Tip vs MCP comparison
        y_diff_mcp = mcp.y - tip.y
        
        if i == 1:  # Index finger
            threshold_mcp = 0.025
        elif i == 2:  # Middle finger
            threshold_mcp = 0.022
        elif i == 3:  # Ring finger
            threshold_mcp = 0.018
        elif i == 4:  # Pinky finger
            threshold_mcp = 0.015
        
        method2_extended = y_diff_mcp > threshold_mcp
        
        # METHOD 3: Relative position analysis
        relative_height = (pip.y + mcp.y) / 2 - tip.y
        relative_threshold = 0.015 if i <= 2 else 0.010
        
        method3_extended = relative_height > relative_threshold
        
        # METHOD 4: Additional strict check for index and middle fingers
        # Check if finger is REALLY straight by comparing multiple joints
        method4_extended = True  # Default to true for ring/pinky
        
        if i <= 2:  # Only for index and middle fingers
            try:
                # Get additional joint points for straighter detection
                dip_joint = hand_landmarks.landmark[tips_ids[i] - 1]  # DIP joint
                
                # Check if the finger forms a relatively straight line
                # Calculate the "straightness" of the finger
                vec_mcp_pip = [pip.x - mcp.x, pip.y - mcp.y]
                vec_pip_dip = [dip_joint.x - pip.x, dip_joint.y - pip.y]
                vec_dip_tip = [tip.x - dip_joint.x, tip.y - dip_joint.y]
                
                # All vectors should point in roughly the same direction for a straight finger
                # Check Y-direction consistency (all should be negative for extended finger)
                y_directions = [vec_mcp_pip[1], vec_pip_dip[1], vec_dip_tip[1]]
                negative_y_count = sum(1 for y in y_directions if y < -0.005)
                
                # For a truly extended finger, at least 2 of 3 segments should point upward
                method4_extended = negative_y_count >= 2
                
                # Additional check: tip should be significantly higher than all other joints
                joints_y = [mcp.y, pip.y, dip_joint.y, tip.y]
                tip_is_highest = tip.y == min(joints_y)
                
                method4_extended = method4_extended and tip_is_highest
                
            except:
                method4_extended = True  # If calculation fails, don't block detection
        
        # DECISION LOGIC: VERY STRICT for index and middle fingers
        votes = sum([method1_extended, method2_extended, method3_extended, method4_extended])
        
        # Index and middle fingers: Require ALL 4 methods to agree (fully extended only)
        # Ring and pinky: Only need 2 out of 3 methods (more lenient, skip method4)
        if i <= 2:  # Index and middle fingers
            finger_extended = votes >= 4  # ALL methods must agree for full extension
        else:  # Ring and pinky
            finger_extended = sum([method1_extended, method2_extended, method3_extended]) >= 2
        
        # CONFIDENCE BOOST: Only for very clear extensions of index/middle
        strong_indicators = []
        if i <= 2:  # Index and middle - very strict confidence boost
            strong_indicators = [
                y_diff_pip > threshold_pip * 2.0,  # MUCH clearer PIP difference
                y_diff_mcp > threshold_mcp * 1.8,  # MUCH clearer MCP difference
                relative_height > relative_threshold * 2.0  # MUCH clearer relative position
            ]
        else:  # Ring and pinky - normal confidence boost
            strong_indicators = [
                y_diff_pip > threshold_pip * 1.5,
                y_diff_mcp > threshold_mcp * 1.3,
                relative_height > relative_threshold * 1.4
            ]
        
        if any(strong_indicators):
            finger_extended = True
        
        states.append('1' if finger_extended else '0')
        
        # Debug output for index and middle fingers
        if i <= 2 and (finger_extended or any([method1_extended, method2_extended, method3_extended])):
            finger_name = "Index" if i == 1 else "Middle"
            print(f"🔍 {finger_name} finger: PIP={y_diff_pip:.3f}({method1_extended}) "
                  f"MCP={y_diff_mcp:.3f}({method2_extended}) "
                  f"REL={relative_height:.3f}({method3_extended}) "
                  f"STR={method4_extended} → {'EXTENDED' if finger_extended else 'NOT EXTENDED'}")

    return ''.join(states)

def calculate_hand_movement(current_landmarks, previous_landmarks):
    if previous_landmarks is None:
        return 0
    
    total_distance = 0
    key_landmarks = [0, 4, 8, 12, 16, 20]  # Wrist and fingertips
    
    for i in key_landmarks:
        curr = current_landmarks.landmark[i]
        prev = previous_landmarks.landmark[i]
        distance = ((curr.x - prev.x) ** 2 + (curr.y - prev.y) ** 2) ** 0.5
        total_distance += distance
    
    return total_distance / len(key_landmarks)

class ASLDetectionServer:
    def __init__(self):
        self.connected_clients = set()
        self.current_state = {
            'current_sign': '00000',
            'sequence_buffer': [],
            'last_action': None,
            'movement': 0.0,
            'is_stable': False,
            'timestamp': time.time(),
        }
        self.action_timeout = 3.0
        self.last_action_time = 0
        
        # High-speed detection components
        self.stabilizer = SignStabilizer(history_size=3, stability_threshold=0.6)  # Smaller buffer for speed
        self.last_stable_sign = None
        self.previous_landmarks = None
        self.movement_buffer = deque(maxlen=5)
        
        # ULTRA-FAST timing controls for 5G-like responsiveness
        self.last_sign_time = 0
        self.min_sign_duration = 0.15  # Super fast: 0.15 seconds hold time
        self.max_movement_threshold = 0.15  # More lenient for speed
        self.wait_between_signs = 0.8  # Very fast: 0.8 seconds between signs
        self.sequence_timeout = 4.0  # Shorter timeout for speed
        
        # State tracking
        self.current_sign_start_time = 0
        self.is_holding_sign = False
        self.sequence_buffer = []
        
        # HIGH-SPEED MediaPipe setup
        self.hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,  # Slightly lower for faster detection
            min_tracking_confidence=0.7,   # Slightly lower for faster tracking
            model_complexity=0             # FASTEST: Use simple model for speed
        )
        
    async def register(self, websocket):
        self.connected_clients.add(websocket)
        client_info = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        print(f"📱 Client connected: {client_info}")
        print(f"👥 Total clients: {len(self.connected_clients)}")
        
        # Send current state to new client
        await websocket.send(json.dumps(self.current_state))
        
    async def unregister(self, websocket):
        self.connected_clients.discard(websocket)
        try:
            client_info = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
            print(f"📱 Client disconnected: {client_info}")
        except:
            print("📱 Client disconnected")
        print(f"👥 Total clients: {len(self.connected_clients)}")
        
    async def broadcast_state(self):
        if not self.connected_clients:
            return
            
        # Clear action after timeout
        current_time = time.time()
        if (self.current_state['last_action'] and 
            current_time - self.last_action_time > self.action_timeout):
            self.current_state['last_action'] = None
        
        self.current_state['timestamp'] = current_time
        message = json.dumps(self.current_state)
        disconnected_clients = []
        
        for client in self.connected_clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.append(client)
            except Exception as e:
                print(f"❌ Error sending to client: {e}")
                disconnected_clients.append(client)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            self.connected_clients.discard(client)
    
    def set_action(self, action):
        """Set an action and record timestamp"""
        self.current_state['last_action'] = action
        self.last_action_time = time.time()
    
    def process_frame(self, frame_data):
        """Process a frame received from Flutter app"""
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(frame_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                print("❌ Failed to decode frame")
                return
            
            # Process the frame
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)
            
            current_time = time.time()
            current_movement = 0.0
            is_stable = False
            current_sign = '00000'
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Validate if this is actually a hand
                    if not is_valid_hand(hand_landmarks):
                        print("🚫 Invalid hand detected, ignoring...")
                        current_sign = '00000'
                        self.previous_landmarks = None
                        self.stabilizer.clear()
                        self.is_holding_sign = False
                        continue
                    
                    # Calculate movement
                    movement = calculate_hand_movement(hand_landmarks, self.previous_landmarks)
                    self.movement_buffer.append(movement)
                    
                    # Use weighted average
                    weights = np.linspace(0.5, 1.0, len(self.movement_buffer))
                    weighted_movements = np.array(self.movement_buffer) * weights
                    avg_movement = np.sum(weighted_movements) / np.sum(weights)
                    current_movement = avg_movement
                    
                    self.previous_landmarks = hand_landmarks
                    
                    # Get finger detection
                    raw_code = get_finger_code_enhanced(hand_landmarks)
                    current_sign = raw_code
                    
                    # Calculate finger count for debug
                    finger_count = sum(int(digit) for digit in raw_code[1:])
                    
                    # Debug output
                    if raw_code != '00000' and raw_code not in IGNORED_SIGNS:
                        print(f"🔍 Detected: {raw_code} ({finger_count} fingers) (movement: {avg_movement:.3f})")
                    
                    # Skip ignored signs
                    if raw_code in IGNORED_SIGNS:
                        self.stabilizer.clear()
                        self.is_holding_sign = False
                        continue
                    
                    # Add to stabilizer
                    self.stabilizer.add_detection(raw_code)
                    
                    # Check stability and process sequences
                    if avg_movement < self.max_movement_threshold:
                        if not self.is_holding_sign:
                            self.current_sign_start_time = current_time
                            self.is_holding_sign = True
                            print(f"👤 Hand stabilized for sign: {raw_code}")
                        
                        is_stable = True
                        
                        # Check if sign has been held long enough
                        if current_time - self.current_sign_start_time >= self.min_sign_duration:
                            stable_code = self.stabilizer.get_stable_sign()
                            
                            if stable_code and stable_code != self.last_stable_sign:
                                # ULTRA-FAST VALIDATION: Minimal delay for speed
                                time_since_last_detection = current_time - getattr(self, 'last_detection_time', 0)
                                if time_since_last_detection < 0.2:  # Very short delay for 5G speed
                                    return
                                
                                self.last_detection_time = current_time
                                
                                # Process sequences
                                self._process_sequence(stable_code, current_time)
                    else:
                        # Hand is moving too much
                        if self.is_holding_sign:
                            print(f"📱 Hand movement detected ({avg_movement:.3f}), waiting for stability...")
                        self.is_holding_sign = False
                        self.stabilizer.clear()
                        is_stable = False
                    
                    # Check for sequence timeout
                    if self.sequence_buffer and current_time - self.last_sign_time > self.sequence_timeout:
                        print(f"⏰ Sequence timed out, clearing buffer: {self.sequence_buffer}")
                        self.sequence_buffer = []
                        self.last_stable_sign = None
            else:
                # No hand detected
                self.previous_landmarks = None
                self.stabilizer.clear()
                self.is_holding_sign = False
                current_sign = '00000'
            
            # Update current state
            self.current_state.update({
                'current_sign': current_sign,
                'sequence_buffer': self.sequence_buffer.copy(),
                'movement': current_movement,
                'is_stable': is_stable,
            })
            
        except Exception as e:
            print(f"❌ Error processing frame: {e}")
    
    def _process_sequence(self, stable_code, current_time):
        """Process sign sequences"""
        if not self.sequence_buffer:
            # First sign
            if stable_code in VALID_FIRST_SIGNS:
                self.sequence_buffer.append(stable_code)
                self.last_stable_sign = stable_code
                self.last_sign_time = current_time
                print(f"🅰 First Sign: {stable_code} - WAIT 0.8 SECONDS for next sign")
                self.stabilizer.clear()
        
        elif len(self.sequence_buffer) == 1:
            # Second sign
            time_since_last = current_time - self.last_sign_time
            if time_since_last >= self.wait_between_signs:
                self.sequence_buffer.append(stable_code)
                self.last_stable_sign = stable_code
                self.last_sign_time = current_time
                print(f"🅱 Second Sign: {stable_code} (after {time_since_last:.1f}s delay)")
                
                # Check for 2-sign sequence
                pair = tuple(self.sequence_buffer)
                action = SEQUENCE_ACTIONS.get(pair)
                if action:
                    print(f"✅ 2-Sign Action: {action}")
                    self.set_action(action)
                    self.sequence_buffer = []
                else:
                    print(f"🔄 Waiting for third sign... Current: {self.sequence_buffer}")
                self.stabilizer.clear()
            else:
                remaining = self.wait_between_signs - time_since_last
                print(f"⏳ Wait {remaining:.1f}s more before next sign (detected: {stable_code})")
        
        elif len(self.sequence_buffer) == 2:
            # Third sign
            time_since_last = current_time - self.last_sign_time
            if time_since_last >= self.wait_between_signs:
                self.sequence_buffer.append(stable_code)
                self.last_stable_sign = stable_code
                print(f"🅲 Third Sign: {stable_code} (after {time_since_last:.1f}s delay)")
                
                # Check for 3-sign sequence
                triple = tuple(self.sequence_buffer)
                action = SEQUENCE_ACTIONS.get(triple)
                if action:
                    print(f"✅ 3-Sign Action: {action}")
                    self.set_action(action)
                else:
                    print(f"❌ Invalid 3-sign sequence: {triple}")
                
                self.sequence_buffer = []
                self.stabilizer.clear()
            else:
                remaining = self.wait_between_signs - time_since_last
                print(f"⏳ Wait {remaining:.1f}s more before third sign (detected: {stable_code})")

# Global server instance
detection_server = ASLDetectionServer()

async def handle_client(websocket):
    await detection_server.register(websocket)
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                
                if data.get('type') == 'frame':
                    # Process frame from Flutter
                    frame_data = data.get('data')
                    if frame_data:
                        detection_server.process_frame(frame_data)
                
                elif data.get('command') == 'ping':
                    await websocket.send(json.dumps({"status": "pong"}))
                    
            except json.JSONDecodeError:
                pass
                
    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception as e:
        print(f"❌ Client handler error: {e}")
    finally:
        await detection_server.unregister(websocket)

async def broadcast_loop():
    """Continuously broadcast detection state to all connected clients"""
    while True:
        try:
            await detection_server.broadcast_state()
            await asyncio.sleep(0.016)  # 60 FPS broadcast rate for ultra-smooth updates
        except Exception as e:
            print(f"❌ Broadcast error: {e}")
            await asyncio.sleep(1)

async def main():
    local_ip = get_local_ip()
    
    print("🚀 Starting ULTRA-HIGH-SPEED Flutter Camera ASL Server...")
    print("=" * 60)
    print(f"🏠 Local IP: {local_ip}")
    print(f"🔗 Local URL: ws://{local_ip}:8765")
    print(f"🔗 Localhost URL: ws://localhost:8765")
    print("=" * 60)
    print("⚡ 5G-LIKE PERFORMANCE MODE:")
    print("   • 30 FPS camera input")
    print("   • 60 FPS server broadcast")
    print("   • 0.15s sign detection")
    print("   • 0.8s between signs")
    print("   • Ultra-low latency processing")
    print("   • Optimized MediaPipe (simple model)")
    print("=" * 60)
    
    # Start WebSocket server
    server = websockets.serve(handle_client, "0.0.0.0", 8765)
    
    print("✅ ULTRA-HIGH-SPEED Server started! Waiting for connections...")
    print("🔥 Performance Mode: 5G-LIKE SPEED ENABLED")
    print("Press Ctrl+C to stop the server")
    
    # Run server and broadcast loop concurrently
    await asyncio.gather(
        server,
        broadcast_loop()
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Shutting down server...")
    except Exception as e:
        print(f"❌ Server error: {e}")

# requirements.txt for the server:
"""
opencv-python==4.8.1.78
mediapipe==0.10.7
websockets==11.0.3
numpy==1.24.3
"""
