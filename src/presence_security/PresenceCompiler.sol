// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract PresenceCompiler {
    enum Level { None, Concept, Practitioner, Studio, Enterprise }

    struct ManifestProof {
        address submitter;
        bytes32 manifestHash; // keccak256(manifest JSON)
        bytes32 assetsHash;   // keccak256(hero||web||disruption)
        uint64  timestamp;
        Level   level;
    }

    mapping(address => Level) public licenses;
    mapping(bytes32 => ManifestProof) public manifests;

    event Licensed(address indexed who, Level level);
    event Compiled(bytes32 indexed manifestId, address indexed who, Level level);

    modifier requiresLicense(Level minLevel) {
        require(uint(licenses[msg.sender]) >= uint(minLevel), "License level too low");
        _;
    }

    function setLicense(address who, Level level) external /* onlyOwner */ {
        licenses[who] = level;
        emit Licensed(who, level);
    }

    function compilePortrait(bytes32 manifestHash, bytes32 assetsHash)
        external requiresLicense(Level.Studio) returns (bytes32 manifestId)
    {
        manifestId = keccak256(abi.encodePacked(manifestHash, assetsHash, msg.sender, block.timestamp));
        manifests[manifestId] = ManifestProof({
            submitter: msg.sender,
            manifestHash: manifestHash,
            assetsHash: assetsHash,
            timestamp: uint64(block.timestamp),
            level: licenses[msg.sender]
        });
        emit Compiled(manifestId, msg.sender, licenses[msg.sender]);
        return manifestId;
    }
}
