#ifndef DIRECT_ICP_OVERLAY_H__
#define DIRECT_ICP_OVERLAY_H__
#include "core/Frame.h"
#include "direct_icp/DirectIcp.h"
namespace vslam
{
class DirectIcpOverlay
{
public:
  struct Entry
  {
    const Frame &f0, f1;
    std::vector<DirectIcp::Feature::ShPtr> constraints;
  };
  void update(const Entry & e);

private:
};
}  // namespace vslam
#endif