// -*-c++-*--------------------------------------------------------------------
// Copyright 2024 Bernd Pfrommer <bernd.pfrommer@gmail.com>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <unistd.h>

#include <chrono>
#ifdef USE_CV_BRIDGE_HPP
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif
#include <ffmpeg_encoder_decoder/decoder.hpp>
#include <ffmpeg_encoder_decoder/encoder.hpp>
#include <ffmpeg_encoder_decoder/utils.hpp>
#include <ffmpeg_image_transport_msgs/msg/ffmpeg_packet.hpp>
#include <ffmpeg_image_transport_tools/bag_processor.hpp>
#include <ffmpeg_image_transport_tools/message_processor.hpp>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <sstream>
#include <unordered_map>

void usage()
{
  std::cout
    << "usage:\n\n"
    << "bag_to_video -i input_bag -t topic -r framerate [options]\n\n"
    << "options:\n"
    << " -o out_file        name of the output file (defaults to \"video\").\n"
    << " -d decoder         name of the libav decoder (hevc_cuvid, libx264 etc).\n"
    << " -e encoder         name of the libav encoder (deprecated, use -E encoder:<name>).\n"
    << " -E key:value       encoder options (can be specified multiple times):\n"
    << "                    encoder:<name_of_encoder> (defaults to libx264)\n"
    << "                    preset:<preset> (p1-p7 for nvenc, slow/medium/fast for x264)\n"
    << "                    cq:<value> (constant quality, 0-51, lower = better)\n"
    << "                    qmax:<value> (max quantization, 1 = best quality)\n"
    << "                    bitrate:<value> (e.g., 8M)\n"
    << "                    rc:<mode> (rate control mode, e.g., vbr, cbr)\n"
    << "                    maxrate:<value> (max bitrate)\n"
    << "                    bufsize:<value> (buffer size)\n"
    << "                    gop_size:<value> (GOP size/keyframe interval)\n"
    << "                    bf:<value> (max B-frames)\n"
    << "                    max_b_frames:<value> (alias for bf)\n"
    << "                    pix_fmt:<format> (pixel format: yuv420p, yuv444p, etc)\n"
    << "                    tune:<value> (tune setting: hq, ll, etc)\n"
    << "                    spatial-aq:<0|1> (spatial adaptive quantization)\n"
    << "                    tier:<value> (encoding tier)\n"
    << "                    profile:<value> (encoding profile)\n"
    << "                    crf:<value> (constant rate factor)\n"
    << "                    delay:<value> (encoding delay)\n"
    << "                    bit_rate:<value> (alias for bitrate)\n"
    << "                    cv_bridge_target_format:<format> (ROS format, e.g., rgb8)\n"
    << "                    av_source_pixel_format:<format> (libav pixel format)\n"
    << "                    measure_performance:<0|1> (enable performance measurement)\n"
    << "                    (any other libav option following key:value syntax)\n"
    << " -r framerate       output framerate (fps).\n"
    << " -T timestamp_file  name of time stamp file.\n"
    << " -s start_time      time in sec since epoch.\n"
    << " --end-time time    end time in sec since epoch.\n"
    << "\nDeprecated options (use -E instead):\n"
    << " -p preset          use -E preset:<value>\n"
    << " -q cq_value        use -E cq:<value>\n"
    << " -b bitrate         use -E bitrate:<value>\n"
    << " -g gop_size        use -E gop_size:<value>\n"
    << " --pix-fmt format   use -E pix_fmt:<value>\n"
    << " --tune tune        use -E tune:<value>\n"
    << " --spatial-aq val   use -E spatial-aq:<value>" << std::endl;
}

using ffmpeg_encoder_decoder::Decoder;
using ffmpeg_encoder_decoder::Encoder;
using ffmpeg_image_transport_msgs::msg::FFMPEGPacket;
using sensor_msgs::msg::Image;
using Path = std::filesystem::path;
using rclcpp::Time;
using bag_time_t = rcutils_time_point_value_t;
using ffmpeg_encoder_decoder::utils::split_by_char;

using namespace std::placeholders;
namespace fs = std::filesystem;

rclcpp::Logger logger = rclcpp::get_logger("bag_to_video");

class VideoWriter : public ffmpeg_image_transport_tools::MessageProcessor<FFMPEGPacket>
{
public:
  VideoWriter(
    const std::vector<std::string> & decoders, const std::string & output_file,
    const std::string & ts_file, const std::string & encoder_name = "libx264",
    const std::map<std::string, std::string> & encoder_options = {}, double framerate = 30.0)
  : output_file_(output_file),
    decoder_names_(decoders),
    encoder_name_(encoder_name),
    encoder_options_(encoder_options),
    framerate_(framerate)
  {
    ts_file_.open(ts_file);
    encoded_ts_file_.open(ts_file + "_encoded");

    // Initialize encoder with options
    for (const auto & [key, value] : encoder_options_) {
      encoder_.addAVOption(key, value);
    }
  }

  void process(
    rcutils_time_point_value_t t_recv, rcutils_time_point_value_t, const std::string &,
    const FFMPEGPacket::ConstSharedPtr & m) final
  {
    if (!decoder_.isInitialized()) {
      if (firstTime_) {
        RCLCPP_INFO_STREAM(logger, "decoding packets for codec: " << m->encoding);
        firstTime_ = false;
        if (decoder_names_.empty()) {
          decoder_names_ =
            split_by_char(Decoder::findDecoders(split_by_char(m->encoding, ';')[0]), ',');
        }
      }
      while (!decoder_names_.empty()) {
        const auto name = *decoder_names_.begin();
        decoder_names_.erase(decoder_names_.begin());  // mark as used
        if (!decoder_.initialize(
              m->encoding, std::bind(&VideoWriter::callback, this, _1, _2, _3), name)) {
          RCLCPP_ERROR_STREAM(
            logger, "cannot initialize decoder " << name << " for encoding: " << m->encoding);
        } else {
          RCLCPP_INFO_STREAM(logger, "using decoder: " << name);
          waitForKeyFrame_ = true;
          break;
        }
      }
      if (!decoder_.isInitialized()) {
        RCLCPP_ERROR_STREAM(logger, "no valid decoder found for encoding: " << m->encoding);
        throw(std::runtime_error("cannot init codec"));
      }
    }
    const size_t current_packet_number = packet_number_++;
    const int64_t stamp_ns = Time(m->header.stamp).nanoseconds();
    const int64_t recv_ns = rclcpp::Time(t_recv).nanoseconds();

    ptsToPacketInfo_[m->pts] = {current_packet_number, stamp_ns, recv_ns};
    if (waitForKeyFrame_) {
      if (!(m->flags & 0x0001)) {
        if (!waitingForKeyFrame_) {
          RCLCPP_INFO_STREAM(logger, "skipping non-key frames starting with pts: " << m->pts);
          waitingForKeyFrame_ = true;
        }
        return;
      }
      if (waitingForKeyFrame_) {
        RCLCPP_INFO_STREAM(logger, "skipped non-key frames until pts: " << m->pts);
        waitingForKeyFrame_ = false;
      }
      waitForKeyFrame_ = false;
    }

    // Store packet info indexed by PTS for later timestamp writing

    const bool ret = decoder_.decodePacket(
      m->encoding, m->data.data(), m->data.size(), m->pts, m->header.frame_id, m->header.stamp);
    if (!ret) {
      decoder_.reset();
    }

    // Get list of successfully decoded PTS values and write their timestamps
    const auto decodedPTS = decoder_.getAndClearDecodedPTS();
    for (const auto & pts : decodedPTS) {
      auto it = ptsToPacketInfo_.find(pts);
      if (it != ptsToPacketInfo_.end()) {
        const auto & info = it->second;
        ts_file_ << info.packet_number << " " << pts << " " << info.stamp_ns << " " << info.recv_ns
                 << std::endl;
        ptsToPacketInfo_.erase(it);  // Remove after writing to prevent memory buildup
      } else {
        RCLCPP_WARN_STREAM(logger, "Could not find packet info for decoded PTS: " << pts);
      }
    }
  }

  void callback(const Image::ConstSharedPtr & msg, bool, const std::string & avPixFmt)
  {
    if (!encoder_.isInitialized()) {
      // Initialize encoder with the first frame
      RCLCPP_INFO_STREAM(logger, "initializing encoder " << encoder_name_);

      auto encode_callback = [this](
                               const std::string & frame_id, const rclcpp::Time & stamp,
                               const std::string & codec, uint32_t width, uint32_t height,
                               uint64_t pts, uint8_t flags, uint8_t * data, size_t sz) {
        this->encodeCallback(frame_id, stamp, codec, width, height, pts, flags, data, sz);
      };

      encoder_.setEncoder(encoder_name_);
      if (!encoder_.initialize(msg->width, msg->height, encode_callback, msg->encoding)) {
        RCLCPP_ERROR_STREAM(logger, "failed to initialize encoder: " << encoder_name_);
        throw std::runtime_error("encoder initialization failed");
      }

      // Open output file based on codec
      codec_ext_ = getCodecExtension();
      raw_file_.open(output_file_ + "." + codec_ext_, std::ios::binary);
      RCLCPP_INFO_STREAM(logger, "writing encoded packets to: " << output_file_ + "." + codec_ext_);
    }

    if (!printedHeader_) {
      RCLCPP_INFO_STREAM(
        logger, "ros encoding: " << msg->encoding << " " << msg->width << "x" << msg->height
                                 << " step: " << msg->step << " libav pix fmt: " << avPixFmt);
      printedHeader_ = true;
    }

    // Encode the frame
    encoder_.encodeImage(*msg);

    if (++frame_number_ % 100 == 0) {
      RCLCPP_INFO_STREAM(logger, "encoded " << frame_number_ << " frames.");
    }
  }

  void encodeCallback(
    const std::string & /*frame_id*/, const rclcpp::Time & stamp, const std::string & /*codec*/,
    uint32_t /*width*/, uint32_t /*height*/, uint64_t pts, uint8_t /*flags*/, uint8_t * data,
    size_t sz)
  {
    // Write encoded packet to file
    raw_file_.write(reinterpret_cast<const char *>(data), sz);

    // Write timestamp info for encoded frames
    encoded_ts_file_ << encoded_frame_number_++ << " " << pts << " " << stamp.nanoseconds() << " "
                     << stamp.nanoseconds() << std::endl;
  }

  const std::string & getOutputFile() const { return output_file_; }

  std::string getCodecExtension() const
  {
    if (encoder_name_.find("264") != std::string::npos) {
      return "h264";
    } else if (encoder_name_.find("av1") != std::string::npos) {
      return "av1";
    }
    return "h265";  // default for HEVC
  }

  void flush()
  {
    if (encoder_.isInitialized()) {
      encoder_.flush();
    }
    raw_file_.close();
    ts_file_.close();
    encoded_ts_file_.close();
  }

private:
  struct PacketInfo
  {
    size_t packet_number;
    int64_t stamp_ns;
    int64_t recv_ns;
  };

  std::unordered_map<uint64_t, PacketInfo> ptsToPacketInfo_;  // Map PTS to packet info
  std::string output_file_;
  std::vector<std::string> decoder_names_;
  std::string encoder_name_;
  std::map<std::string, std::string> encoder_options_;
  double framerate_;
  std::string codec_ext_;
  std::ofstream ts_file_;
  std::ofstream raw_file_;
  std::ofstream encoded_ts_file_;
  size_t frame_number_{0};
  size_t encoded_frame_number_{0};
  Decoder decoder_;
  Encoder encoder_;
  size_t packet_number_{0};
  bool firstTime_{true};
  bool waitForKeyFrame_{false};
  bool waitingForKeyFrame_{false};
  bool printedHeader_{false};
};

static void convertToMP4(
  const std::string & raw_file, const std::string & mp4_file, const std::string & codec_ext,
  double rate)
{
  std::stringstream ss;
  ss << "ffmpeg -y -err_detect ignore_err -fflags +genpts -r " << rate << " -i " << raw_file << "."
     << codec_ext << " -c:v copy " << mp4_file << ".mp4";

  RCLCPP_INFO_STREAM(logger, "Converting to MP4: " << ss.str());
  int rc = std::system(ss.str().c_str());
  if (rc == -1 || rc != 0) {
    RCLCPP_ERROR_STREAM(logger, "Failed to convert to MP4: " << ss.str());
  } else {
    // Clean up raw file
    rc = std::system(("rm " + raw_file + "." + codec_ext).c_str());
    if (rc == -1) {
      RCLCPP_ERROR_STREAM(logger, "Failed to remove raw file: " << raw_file << "." << codec_ext);
    }
  }
}

// Helper function to parse key:value pairs
std::pair<std::string, std::string> parseKeyValue(const std::string & input)
{
  size_t colon_pos = input.find(':');
  if (colon_pos == std::string::npos) {
    throw std::runtime_error("Invalid key:value format: " + input);
  }
  std::string key = input.substr(0, colon_pos);
  std::string value = input.substr(colon_pos + 1);
  return {key, value};
}

int main(int argc, char ** argv)
{
  int opt;
  std::string bag;
  std::string out_file = "video";
  std::string topic;
  std::string time_stamp_file = "timestamps.txt";
  std::string decoder;
  std::string encoder = "libx264";  // Default encoder
  std::string framerate = "30";

  // Map to store all encoder options
  std::map<std::string, std::string> encoder_options;

  bag_time_t start_time = std::numeric_limits<bag_time_t>::min();
  bag_time_t end_time = std::numeric_limits<bag_time_t>::max();

  while ((opt = getopt(argc, argv, "i:d:e:E:p:q:b:r:g:o:s:t:T:h")) != -1) {
    switch (opt) {
      case 'i':
        bag = optarg;
        break;
      case 'd':
        decoder = optarg;
        break;
      case 'e':
        encoder = optarg;
        RCLCPP_WARN(logger, "Option -e is deprecated, use -E encoder:<name> instead");
        break;
      case 'E': {
        try {
          auto [key, value] = parseKeyValue(optarg);

          // Special handling for encoder name
          if (key == "encoder") {
            encoder = value;
          } else if (key == "bit_rate") {
            // Map bit_rate to b:v for compatibility
            encoder_options["b:v"] = value;
          } else if (key == "bitrate") {
            encoder_options["maxrate"] = value;
          } else if (key == "max_b_frames") {
            encoder_options["bf"] = value;
          } else {
            encoder_options[key] = value;
          }

          RCLCPP_INFO_STREAM(logger, "Encoder option: " << key << " = " << value);
        } catch (const std::exception & e) {
          std::cerr << "Error parsing -E option: " << e.what() << std::endl;
          usage();
          return -1;
        }
        break;
      }
      case 'p':
        encoder_options["preset"] = optarg;
        RCLCPP_WARN(logger, "Option -p is deprecated, use -E preset:<value> instead");
        break;
      case 'q':
        encoder_options["cq"] = optarg;
        RCLCPP_WARN(logger, "Option -q is deprecated, use -E cq:<value> instead");
        break;
      case 'b':
        encoder_options["maxrate"] = optarg;
        RCLCPP_WARN(logger, "Option -b is deprecated, use -E bitrate:<value> instead");
        break;
      case 'r':
        framerate = optarg;
        break;
      case 'g':
        encoder_options["g"] = optarg;
        RCLCPP_WARN(logger, "Option -g is deprecated, use -E gop_size:<value> instead");
        break;
      case 'o':
        out_file = optarg;
        break;
      case 's':
        start_time = static_cast<bag_time_t>(atof(optarg) * 1e9);
        if (start_time < 0) {
          std::cout << "start time out of range, must be in seconds since start of epoch"
                    << std::endl;
          usage();
          return (-1);
        }
        break;
      case 't':
        topic = optarg;
        break;
      case 'T':
        time_stamp_file = optarg;
        break;
      case 'h':
        usage();
        return (-1);
        break;
      default:
        std::cout << "unknown option: " << opt << std::endl;
        usage();
        return (-1);
        break;
    }
  }

  if (bag.empty()) {
    std::cout << "missing bag file argument!" << std::endl;
    usage();
    return (-1);
  }
  if (topic.empty()) {
    std::cout << "missing topic argument!" << std::endl;
    usage();
    return (-1);
  }

  // Print final encoder configuration
  RCLCPP_INFO_STREAM(logger, "Using encoder: " << encoder);
  if (!encoder_options.empty()) {
    RCLCPP_INFO(logger, "Encoder options:");
    for (const auto & [key, value] : encoder_options) {
      RCLCPP_INFO_STREAM(logger, "  " << key << ": " << value);
    }
  }

  const std::vector<std::string> topics{topic};
  const std::string topic_type = "ffmpeg_image_transport_msgs/msg/FFMPEGPacket";
  ffmpeg_image_transport_tools::BagProcessor<FFMPEGPacket> bproc(
    logger, bag, topics, topic_type, start_time, end_time);

  std::vector<std::string> decoders;
  if (!decoder.empty()) {
    decoders.push_back(decoder);
  }

  VideoWriter vw(
    decoders, out_file, time_stamp_file, encoder, encoder_options, std::stod(framerate));
  bproc.process(&vw);

  // Flush the encoder and close files
  vw.flush();

  // Convert raw encoded file to MP4
  convertToMP4(vw.getOutputFile(), out_file, vw.getCodecExtension(), std::stod(framerate));

  return 0;
}