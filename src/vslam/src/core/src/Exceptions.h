// Copyright 2022 Philipp.Duernay
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

//
// Created by phil on 06.07.21.
//

#ifndef DIRECT_IMAGE_ALIGNMENT_EXCEPTIONS_H
#define DIRECT_IMAGE_ALIGNMENT_EXCEPTIONS_H

#include <stdexcept>
#include <string>
namespace pd
{
class Exception : public std::runtime_error
{
public:
  Exception(const std::string & msg) : std::runtime_error(msg) {}
};
}  // namespace pd
#endif  //DIRECT_IMAGE_ALIGNMENT_EXCEPTIONS_H
