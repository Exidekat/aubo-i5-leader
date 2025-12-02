# AUBO i5 Leader-Follower Control System

A leader-follower robot control system that reads joint angles from a Mini AUBO i5 leader arm (built with Feetech servos) and mirrors them directly to control a real AUBO i5 follower arm.

## Features

- Direct 1:1 joint angle mapping from Mini AUBO leader to AUBO i5 follower
- Real-time control via Feetech serial protocol (leader) and libpyauboi5 SDK (follower)
- Automatic leader arm calibration with home position capture
- Fake mode for testing without real hardware
- Configurable serial port selection
- Real-time control loop with safety limits

## Installation

### Requirements
- Python 3.11+
- Conda environment: `ltx` (or equivalent)
- AUBO i5 robot controller software running on localhost or network

### Setup

```bash
# Activate the ltx conda environment
conda activate ltx

# Install dependencies
pip install -r requirements.txt

# Key dependencies:
# - pyserial>=3.5          # Feetech servo communication
# - libpyauboi5            # AUBO i5 robot control SDK
```

## Usage

### Running the Leader-Follower System

The main script supports both real hardware and fake mode for testing.

#### Real Mode (with Hardware)

```bash
# Run with real AUBO i5 connection
python main.py --real

# Or specify serial port explicitly
python main.py --real --dev /dev/ttyACM0
```

The script will:
1. Auto-detect or use specified Feetech servo serial port
2. Calibrate the leader arm by saving home positions for all 6 motors
3. Connect to AUBO i5 robot controller (localhost:8899 by default)
4. Read leader joint angles continuously
5. Convert Feetech positions to AUBO joint angles
6. Send commands to follower robot in real-time

#### Fake Mode (Testing without Hardware)

```bash
# Run in fake mode (no robot connection)
python main.py --fake

# Or with specific serial port for leader only
python main.py --fake --dev /dev/ttyACM0
```

In fake mode:
- Leader angles are read and displayed (if connected)
- No commands are sent to the AUBO i5
- Useful for testing leader arm calibration and angle conversion

#### Interactive Mode

```bash
# Run without flags - will prompt you
python main.py

# You'll be asked:
# "Start a real connection to the Aubo i5? (y/n):"
```

### Command-Line Arguments

- `--real`: Connect to real AUBO i5 robot
- `--fake`: Run in fake mode (no robot connection)
- `--dev <port>`: Specify serial port (e.g., `/dev/ttyACM0`, `COM3`)

### Configuration

The script uses these default settings:
- **AUBO i5 IP**: `localhost` (modify in code if using network connection)
- **AUBO i5 Port**: `8899` (default controller port)
- **Feetech Baud Rate**: `1000000` (1 Mbps)
- **Control Rate**: ~100 Hz (10ms loop delay)
- **Conversion Factor**: `2*pi / 4096` (Feetech counts to radians)

## Hardware Specifications

### Mini AUBO i5 Leader Arm (Feetech Servos)
- **DOF**: 6 revolute joints
- **Servos**: Feetech STS3215 smart servos
- **Position Resolution**: 4096 counts per revolution (14-bit)
- **Communication**: Serial protocol via USB (1 Mbps baud)
- **Joint Mapping**: Motors 1-6 correspond directly to AUBO joints 1-6
- **Calibration**: Home position saved on startup

### AUBO i5 Follower Arm
- **DOF**: 6 revolute joints
- **Payload**: 5 kg
- **Max Reach**: 886.5 mm
- **Repeatability**: +/-0.05 mm
- **Joint Limits**: +/-175 degrees (all joints)
- **Control Interface**: libpyauboi5 SDK via TCP/IP (port 8899)

## Control Architecture

### Delta-Based Control Loop

The system uses a simple delta-based approach for direct joint angle mapping:

```python
while True:
    # 1. Read current leader positions (Feetech serial)
    leader_current_counts = read_all_motor_positions()  # Raw 0-4096

    # 2. Calculate delta from home position
    leader_delta_counts = leader_current_counts - leader_home_counts

    # 3. Convert delta to radians
    leader_delta_rad = leader_delta_counts * (2*pi / 4096)

    # 4. Apply delta to follower (1:1 mapping)
    follower_target = leader_delta_rad  # Direct mapping

    # 5. Send to AUBO i5 follower
    robot.move_joint(follower_target, blocking=True)

    time.sleep(0.01)  # ~100 Hz control loop
```

### Key Functions

**`calibrate_leader(port_name)`**
- Reads and saves home position for all 6 Feetech motors
- Stores raw counts (0-4096) in global `leader_home_values` array
- Called once on startup

**`calculate_aubo_angles_from_leader(ser)`**
- Reads current positions from all 6 motors
- Calculates delta: `current - home`
- Converts to radians: `delta * CONVERSION_FACTOR`
- Returns 6-element tuple of joint angles

**`read_motor_position(ser, motor_id)`**
- Sends Feetech read command to specific motor
- Address: 0x38 (position register)
- Returns 14-bit position value (0-4095)

**`test_process_demo(real_connection, com_port)`**
- Main control loop
- Connects to AUBO i5 if `real_connection=True`
- Reads leader angles and sends to follower continuously
- Handles Ctrl+C graceful shutdown

## System Workflow

### Startup Sequence
1. Script starts and prompts for mode (`--real` or `--fake`)
2. Auto-detects or uses specified serial port for leader arm
3. **Calibration Phase**: Reads all 6 motor positions and saves as home
4. Connects to AUBO i5 controller (real mode only)
5. Enters main control loop

### Control Loop (Real Mode)
1. Read current leader motor positions
2. Calculate delta from calibrated home
3. Convert delta to radians
4. Send joint command to AUBO i5
5. Repeat at ~100 Hz

### Control Loop (Fake Mode)
1. Read current leader motor positions
2. Calculate and display angles
3. No commands sent to follower
4. Useful for testing leader arm setup

## Troubleshooting

### Leader Arm Connection Issues

**Problem**: Cannot find Feetech serial port
```bash
# List available ports
ls /dev/ttyACM* /dev/ttyUSB*

# Check permissions
sudo chmod 666 /dev/ttyACM0

# Specify port manually
python main.py --real --dev /dev/ttyACM0
```

**Problem**: Calibration fails or returns invalid values
- Ensure all 6 servos are powered
- Check that servo IDs are 1-6
- Verify baud rate is 1000000 (1 Mbps)
- Try repowering the servos

### AUBO i5 Connection Issues

**Problem**: Cannot connect to AUBO controller
```bash
# Check if controller software is running
# Default: localhost:8899

# If using network connection, modify IP in code:
# Line ~2835: ip = '192.168.65.131'  # Change this
```

**Problem**: `libpyauboi5 module not found`
```bash
# Install AUBO SDK
pip install libpyauboi5

# Or check conda environment
conda activate ltx
pip install libpyauboi5
```

### Runtime Issues

**Problem**: Robot moves erratically
- Ensure proper calibration (leader arm in neutral pose at start)
- Check that all 6 motors are reading correctly
- Verify joint angle conversions are correct

**Problem**: Script freezes or crashes
- Press Ctrl+C for graceful shutdown
- Check log files in `./logfiles/robot-ctl-python.log`
- Restart AUBO controller if needed

## Safety Considerations

- Always start with both arms in neutral/home position for calibration
- Keep emergency stop accessible during operation
- Test in fake mode first before enabling real robot connection
- Be aware of joint limits (+/-175 degrees)
- Monitor for unexpected movements during initial testing
- The control loop runs at ~100 Hz - movements can be fast

## Logs

Log files are stored in `./logfiles/robot-ctl-python.log` with:
- Timestamped events
- Connection status
- Joint angle commands
- Error messages

Log rotation is automatic (50 MB per file, 30 backups).

## References

- **AUBO i5 SDK**: libpyauboi5 (official AUBO Robotics SDK)
- **Feetech Protocol**: STS series smart servo documentation
- **Serial Communication**: pyserial library

---

**Project Status**: Active development for Mini AUBO i5 leader-follower control